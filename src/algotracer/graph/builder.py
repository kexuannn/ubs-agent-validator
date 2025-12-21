from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import sys

from gqlalchemy import Memgraph

from algotracer.analysis.side_effects import infer_side_effects
from algotracer.ingest.ast_parser import ClassInfo, FunctionInfo, ModuleInfo, CallInfo


@dataclass(frozen=True)
class ModuleRecord:
    name: str
    path: str  # stable path (relative to repo_root)
    abs_path: str
    functions: set[str]
    classes: set[str]
    class_methods: Dict[str, set[str]]
    class_bases: Dict[str, List[str]]


@dataclass(frozen=True)
class ImportTarget:
    kind: str  # "module" or "symbol"
    module: str
    symbol: str | None


@dataclass(frozen=True)
class ResolvedTarget:
    kind: str  # "function" or "external"
    module: str | None
    qualname: str | None
    external_name: str | None


@dataclass(frozen=True)
class GraphBuildConfig:
    repo_id: str
    repo_root: Path


def module_name_for_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except Exception:
        rel = path.name
    parts = list(rel.parts) if hasattr(rel, "parts") else [str(rel)]
    if not parts:
        return ""
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join([p for p in parts if p])


def _stable_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except Exception:
        h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
        return f"{path.name}.{h}"


def _stable_module_id(stable_path: str) -> str:
    return f"stable:mod:{stable_path}"


def _stable_function_id(stable_path: str, qualname: str) -> str:
    return f"stable:sym:{stable_path}:{qualname}"


def _stable_class_id(stable_path: str, qualname: str) -> str:
    return f"stable:cls:{stable_path}:{qualname}"


def build_module_records(modules: Sequence[ModuleInfo], repo_root: Path) -> Dict[str, ModuleRecord]:
    records: Dict[str, ModuleRecord] = {}
    for module in modules:
        name = module_name_for_path(module.path, repo_root)
        stable_path = _stable_path(module.path, repo_root)
        functions = {f.name for f in module.functions}
        classes = {c.name for c in module.classes}
        class_methods = {c.name: {m.name for m in c.methods} for c in module.classes}
        class_bases = {c.name: list(c.bases) for c in module.classes}
        records[name] = ModuleRecord(
            name=name,
            path=stable_path,
            abs_path=str(module.path.resolve()),
            functions=functions,
            classes=classes,
            class_methods=class_methods,
            class_bases=class_bases,
        )
    return records


def _resolve_relative_module(current_module: str, raw_module: str | None) -> str | None:
    if raw_module is None:
        return None
    if not raw_module.startswith("."):
        return raw_module
    dots = len(raw_module) - len(raw_module.lstrip("."))
    suffix = raw_module.lstrip(".")
    base_parts = current_module.split(".") if current_module else []
    if dots > len(base_parts):
        return suffix or None
    parent = base_parts[: len(base_parts) - dots]
    if suffix:
        parent.append(suffix)
    return ".".join([p for p in parent if p]) or None


def build_import_map(
    module: ModuleInfo,
    module_name: str,
    module_index: Dict[str, ModuleRecord],
) -> Dict[str, ImportTarget]:
    mapping: Dict[str, ImportTarget] = {}

    for imp in module.imports:
        if imp.kind == "import":
            if imp.alias:
                mapping[imp.alias] = ImportTarget(kind="module", module=imp.name, symbol=None)
            else:
                root = imp.name.split(".")[0]
                mapping[root] = ImportTarget(kind="module", module=root, symbol=None)
            continue

        resolved_module = _resolve_relative_module(module_name, imp.module)
        if not resolved_module:
            continue

        local_name = imp.alias or imp.name
        module_candidate = f"{resolved_module}.{imp.name}" if resolved_module else imp.name
        if module_candidate in module_index:
            mapping[local_name] = ImportTarget(kind="module", module=module_candidate, symbol=None)
        else:
            mapping[local_name] = ImportTarget(kind="symbol", module=resolved_module, symbol=imp.name)

    return mapping


def _resolve_via_imports(
    call: str,
    import_map: Dict[str, ImportTarget],
    module_index: Dict[str, ModuleRecord],
) -> ResolvedTarget | None:
    parts = call.split(".")
    if not parts:
        return None

    if len(parts) == 1:
        target = import_map.get(call)
        if target and target.kind == "symbol" and target.symbol:
            mod = module_index.get(target.module)
            if mod and target.symbol in mod.functions:
                return ResolvedTarget(kind="function", module=target.module, qualname=target.symbol, external_name=None)
        return None

    head = parts[0]
    target = import_map.get(head)
    if not target or target.kind != "module":
        return None

    base_module = target.module
    remaining = parts[1:]

    for i in range(len(remaining), 0, -1):
        candidate_module = base_module + "." + ".".join(remaining[:i])
        if candidate_module in module_index:
            mod = module_index[candidate_module]
            tail = remaining[i:]
            if len(tail) == 1 and tail[0] in mod.functions:
                return ResolvedTarget(kind="function", module=candidate_module, qualname=tail[0], external_name=None)
            if len(tail) == 2 and tail[0] in mod.class_methods and tail[1] in mod.class_methods.get(tail[0], set()):
                return ResolvedTarget(kind="function", module=candidate_module, qualname=f"{tail[0]}.{tail[1]}", external_name=None)

    mod = module_index.get(base_module)
    if not mod:
        return None

    if len(remaining) == 1 and remaining[0] in mod.functions:
        return ResolvedTarget(kind="function", module=base_module, qualname=remaining[0], external_name=None)

    if len(remaining) == 2 and remaining[0] in mod.class_methods and remaining[1] in mod.class_methods.get(remaining[0], set()):
        return ResolvedTarget(kind="function", module=base_module, qualname=f"{remaining[0]}.{remaining[1]}", external_name=None)

    return None


# ---------------------------------------------------------------------------
# NEW: re-export/package fallback
# ---------------------------------------------------------------------------
def _find_class_in_package(
    module_prefix: str,
    class_name: str,
    module_index: Dict[str, ModuleRecord],
) -> Tuple[ModuleRecord | None, str | None]:
    """
    Best-effort fallback for package-level imports / re-exports.

    Example:
        from vnpy.alpha import AlphaStrategy
    where the actual class lives in:
        vnpy.alpha.strategy.template.AlphaStrategy

    We scan modules whose name == module_prefix OR startswith(module_prefix + ".")
    and return the first one that defines the class.
    """
    pref = module_prefix + "."
    for mod_name, mod in module_index.items():
        if mod_name == module_prefix or mod_name.startswith(pref):
            if class_name in mod.class_methods:
                return mod, class_name
    return None, None


def _resolve_base_method(
    base: str,
    method: str,
    *,
    module_record: ModuleRecord,
    module_index: Dict[str, ModuleRecord],
    import_map: Dict[str, ImportTarget],
) -> ResolvedTarget | None:
    if base in module_record.class_methods and method in module_record.class_methods.get(base, set()):
        return ResolvedTarget(kind="function", module=module_record.name, qualname=f"{base}.{method}", external_name=None)

    if "." in base:
        mod_name, class_name = base.rsplit(".", 1)
        mod = module_index.get(mod_name)
        if mod and method in mod.class_methods.get(class_name, set()):
            return ResolvedTarget(kind="function", module=mod.name, qualname=f"{class_name}.{method}", external_name=None)

    target = import_map.get(base)
    if target and target.kind == "symbol" and target.module:
        mod = module_index.get(target.module)
        class_name = target.symbol or base

        # direct hit: module points to defining module
        if mod and method in mod.class_methods.get(class_name, set()):
            return ResolvedTarget(kind="function", module=mod.name, qualname=f"{class_name}.{method}", external_name=None)

        # NEW: fallback for package re-exports (vnpy.alpha -> vnpy.alpha.*)
        mod2, cls2 = _find_class_in_package(target.module, class_name, module_index)
        if mod2 and cls2 and method in mod2.class_methods.get(cls2, set()):
            return ResolvedTarget(kind="function", module=mod2.name, qualname=f"{cls2}.{method}", external_name=None)

    return None


def _locate_base_class(
    base: str,
    *,
    module_record: ModuleRecord,
    module_index: Dict[str, ModuleRecord],
    import_map: Dict[str, ImportTarget],
) -> Tuple[ModuleRecord | None, str | None]:
    if base in module_record.class_methods:
        return module_record, base

    if "." in base:
        mod_name, cls_name = base.rsplit(".", 1)
        mod = module_index.get(mod_name)
        if mod and cls_name in mod.class_methods:
            return mod, cls_name

    target = import_map.get(base)
    if target and target.kind == "symbol" and target.module:
        mod = module_index.get(target.module)
        cls_name = target.symbol or base

        # direct hit
        if mod and cls_name in mod.class_methods:
            return mod, cls_name

        # NEW: fallback for package re-exports
        mod2, cls2 = _find_class_in_package(target.module, cls_name, module_index)
        if mod2 and cls2:
            return mod2, cls2

    return None, None


def _resolve_in_bases(
    method: str,
    current_class: str,
    *,
    module_record: ModuleRecord,
    module_index: Dict[str, ModuleRecord],
    import_map: Dict[str, ImportTarget],
    import_maps: Dict[str, Dict[str, ImportTarget]],
    visited: set[tuple[str, str]] | None = None,
) -> ResolvedTarget | None:
    visited = visited or set()
    visited.add((module_record.name, current_class))

    for base in module_record.class_bases.get(current_class, []):
        candidate = _resolve_base_method(
            base,
            method,
            module_record=module_record,
            module_index=module_index,
            import_map=import_map,
        )
        if candidate is not None:
            return candidate

        base_mod, base_name = _locate_base_class(
            base,
            module_record=module_record,
            module_index=module_index,
            import_map=import_map,
        )
        if not base_mod or not base_name:
            continue

        key = (base_mod.name, base_name)
        if key in visited:
            continue

        base_import_map = import_maps.get(base_mod.name, {})
        deeper = _resolve_in_bases(
            method,
            base_name,
            module_record=base_mod,
            module_index=module_index,
            import_map=base_import_map,
            import_maps=import_maps,
            visited=visited,
        )
        if deeper is not None:
            return deeper

    return None


def _resolve_in_module(
    call: str,
    current_class: str | None,
    module_record: ModuleRecord,
    module_index: Dict[str, ModuleRecord],
    import_maps: Dict[str, Dict[str, ImportTarget]],
) -> ResolvedTarget | None:
    import_map = import_maps.get(module_record.name, {})

    if current_class:
        parts = call.split(".")

        # Handle super().foo(...) → resolve in base classes
        if parts and parts[0] == "super()" and len(parts) >= 2:
            method_name = parts[-1]
            return _resolve_in_bases(
                method_name,
                current_class,
                module_record=module_record,
                module_index=module_index,
                import_map=import_map,
                import_maps=import_maps,
            )

        # Handle direct self/cls method calls ONLY for `self.foo` / `cls.foo`
        # Do NOT treat `self.x.foo` as a method on the current class to avoid
        # bogus self-loops like LassoModel.predict -> LassoModel.predict.
        if parts and parts[0] in {"self", "cls"}:
            # `self.foo` / `cls.foo` → len == 2 → method_name = parts[1]
            if len(parts) == 2:
                method_name = parts[1]
                if method_name in module_record.class_methods.get(current_class, set()):
                    return ResolvedTarget(
                        kind="function",
                        module=module_record.name,
                        qualname=f"{current_class}.{method_name}",
                        external_name=None,
                    )

                candidate = _resolve_in_bases(
                    method_name,
                    current_class,
                    module_record=module_record,
                    module_index=module_index,
                    import_map=import_map,
                    import_maps=import_maps,
                )
                if candidate is not None:
                    return candidate
            # For deeper chains like `self.model.predict`, skip resolving to the
            # current class; these will fall through and be treated as external.

        # Unqualified method calls inside a class: `foo(...)`
        if len(parts) == 1 and call in module_record.class_methods.get(current_class, set()):
            return ResolvedTarget(
                kind="function",
                module=module_record.name,
                qualname=f"{current_class}.{call}",
                external_name=None,
            )

        if len(parts) == 1:
            candidate = _resolve_in_bases(
                call,
                current_class,
                module_record=module_record,
                module_index=module_index,
                import_map=import_map,
                import_maps=import_maps,
            )
            if candidate is not None:
                return candidate

    parts = call.split(".")
    if len(parts) >= 2:
        class_name, method_name = parts[-2], parts[-1]
        if class_name in module_record.class_methods and method_name in module_record.class_methods.get(class_name, set()):
            return ResolvedTarget(
                kind="function",
                module=module_record.name,
                qualname=f"{class_name}.{method_name}",
                external_name=None,
            )

    if call in module_record.functions:
        return ResolvedTarget(
            kind="function",
            module=module_record.name,
            qualname=call,
            external_name=None,
        )

    return None


def resolve_call_target(
    call: str,
    *,
    current_class: str | None,
    module_record: ModuleRecord,
    import_maps: Dict[str, Dict[str, ImportTarget]],
    module_index: Dict[str, ModuleRecord],
) -> ResolvedTarget:
    local = _resolve_in_module(call, current_class, module_record, module_index, import_maps)
    if local is not None:
        return local

    import_map = import_maps.get(module_record.name, {})
    via_imports = _resolve_via_imports(call, import_map, module_index)
    if via_imports is not None:
        return via_imports

    return ResolvedTarget(kind="external", module=None, qualname=None, external_name=call)


def _iter_functions(module: ModuleInfo) -> Iterable[tuple[FunctionInfo, ClassInfo | None]]:
    for func in module.functions:
        yield func, None
    for cls in module.classes:
        for method in cls.methods:
            yield method, cls


def _namespace_for_external(name: str) -> str:
    return name.split(".", 1)[0] if name else "unknown"


def _virtual_call_id(src_func_id: str, callee_attr: str | None, receiver: str | None, lineno: int | None) -> str:
    suffix = "|".join([callee_attr or "unknown", receiver or "unknown", str(lineno or 0)])
    h = hashlib.sha1(f"{src_func_id}:{suffix}".encode("utf-8")).hexdigest()[:12]
    return f"vc:{h}"


_STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", ())) or set(sys.builtin_module_names)


def _classify_external(
    external_name: str,
    *,
    module_record: ModuleRecord,
    import_map: Dict[str, ImportTarget],
) -> str:
    """Heuristic external classification."""
    root = (external_name or "").split(".", 1)[0]
    if root in _STDLIB_MODULES:
        return "stdlib"
    if root in import_map:
        return "third_party"
    return "unresolved"


def _emit_overrides_edges(
    *,
    mg: Memgraph,
    modules: Sequence[ModuleInfo],
    config: GraphBuildConfig,
    repo_root: Path,
    module_index: Dict[str, ModuleRecord],
    module_names_by_path: Dict[str, str],
    import_maps: Dict[str, Dict[str, ImportTarget]],
) -> None:
    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        module_name = module_names_by_path[stable_path]
        module_record = module_index[module_name]
        import_map = import_maps.get(module_record.name, {})

        for cls in module.classes:
            if not module_record.class_bases.get(cls.name):
                continue

            for method in cls.methods:
                base = _resolve_in_bases(
                    method.name,
                    cls.name,
                    module_record=module_record,
                    module_index=module_index,
                    import_map=import_map,
                    import_maps=import_maps,
                )
                if not (base and base.kind == "function" and base.module and base.qualname):
                    continue

                child_id = _stable_function_id(stable_path, f"{cls.name}.{method.name}")

                base_mod = module_index.get(base.module)
                if not base_mod:
                    continue
                parent_id = _stable_function_id(base_mod.path, base.qualname)

                mg.execute(
                    "MATCH (child:Function {repo_id: $repo_id, id: $child_id}), "
                    "(parent:Function {repo_id: $repo_id, id: $parent_id}) "
                    "MERGE (child)-[:OVERRIDES {repo_id: $repo_id}]->(parent)",
                    {"repo_id": config.repo_id, "child_id": child_id, "parent_id": parent_id},
                )


def write_graph(*, mg: Memgraph, modules: Sequence[ModuleInfo], config: GraphBuildConfig) -> None:
    repo_root = config.repo_root
    module_index = build_module_records(modules, repo_root)

    module_names_by_path = {_stable_path(m.path, repo_root): module_name_for_path(m.path, repo_root) for m in modules}

    import_maps: Dict[str, Dict[str, ImportTarget]] = {}
    for module in modules:
        name = module_name_for_path(module.path, repo_root)
        import_maps[name] = build_import_map(module, name, module_index)

    mg.execute(
        "MERGE (r:Repo {repo_id: $repo_id}) SET r.root = $root",
        {"repo_id": config.repo_id, "root": str(repo_root.resolve())},
    )

    # --- modules/classes/functions ---
    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        module_name = module_names_by_path[stable_path]
        module_id = _stable_module_id(stable_path)

        mg.execute(
            "MERGE (m:Module {repo_id: $repo_id, id: $id}) "
            "SET m.path = $path, m.name = $name, m.abs_path = $abs_path",
            {
                "repo_id": config.repo_id,
                "id": module_id,
                "path": stable_path,
                "name": module_name,
                "abs_path": str(module.path.resolve()),
            },
        )
        mg.execute(
            "MATCH (r:Repo {repo_id: $repo_id}), (m:Module {repo_id: $repo_id, id: $module_id}) "
            "MERGE (r)-[:HAS_MODULE {repo_id: $repo_id}]->(m)",
            {"repo_id": config.repo_id, "module_id": module_id},
        )

        for cls in module.classes:
            class_id = _stable_class_id(stable_path, cls.name)
            mg.execute(
                "MERGE (c:Class {repo_id: $repo_id, id: $id}) "
                "SET c.name = $name, c.qualname = $qualname, c.path = $path, c.lineno = $lineno",
                {
                    "repo_id": config.repo_id,
                    "id": class_id,
                    "name": cls.name,
                    "qualname": cls.name,
                    "path": stable_path,
                    "lineno": cls.lineno,
                },
            )
            mg.execute(
                "MATCH (m:Module {repo_id: $repo_id, id: $module_id}), (c:Class {repo_id: $repo_id, id: $class_id}) "
                "MERGE (m)-[:DEFINES_CLASS {repo_id: $repo_id}]->(c)",
                {"repo_id": config.repo_id, "module_id": module_id, "class_id": class_id},
            )

        # Class inheritance edges
        for cls in module.classes:
            child_id = _stable_class_id(stable_path, cls.name)
            for base in cls.bases:
                base_mod, base_name = _locate_base_class(
                    base,
                    module_record=module_index[module_name],
                    module_index=module_index,
                    import_map=import_maps.get(module_name, {}),
                )
                if not base_mod or not base_name:
                    continue
                parent_id = _stable_class_id(base_mod.path, base_name)
                mg.execute(
                    "MATCH (child:Class {repo_id: $repo_id, id: $child_id}), (parent:Class {repo_id: $repo_id, id: $parent_id}) "
                    "MERGE (child)-[:SUBCLASS_OF {repo_id: $repo_id}]->(parent)",
                    {"repo_id": config.repo_id, "child_id": child_id, "parent_id": parent_id},
                )

        for func, cls in _iter_functions(module):
            qualname = f"{cls.name}.{func.name}" if cls is not None else func.name
            func_id = _stable_function_id(stable_path, qualname)
            kind = "method" if cls is not None else "function"
            owner_class_id = _stable_class_id(stable_path, cls.name) if cls is not None else None

            mg.execute(
                "MERGE (f:Function {repo_id: $repo_id, id: $id}) "
                "SET f.name = $name, f.qualname = $qualname, f.path = $path, f.lineno = $lineno, "
                "f.kind = $kind, f.signature = $signature, f.owner_class = $owner_class, f.owner_class_id = $owner_class_id",
                {
                    "repo_id": config.repo_id,
                    "id": func_id,
                    "name": func.name,
                    "qualname": qualname,
                    "path": stable_path,
                    "lineno": func.lineno,
                    "kind": kind,
                    "signature": func.signature,
                    "owner_class": cls.name if cls is not None else None,
                    "owner_class_id": owner_class_id,
                },
            )

            mg.execute(
                "MATCH (m:Module {repo_id: $repo_id, id: $module_id}), (f:Function {repo_id: $repo_id, id: $func_id}) "
                "MERGE (m)-[:DEFINES {repo_id: $repo_id}]->(f)",
                {"repo_id": config.repo_id, "module_id": module_id, "func_id": func_id},
            )

            if cls is not None:
                class_id = _stable_class_id(stable_path, cls.name)
                mg.execute(
                    "MATCH (c:Class {repo_id: $repo_id, id: $class_id}), (f:Function {repo_id: $repo_id, id: $func_id}) "
                    "MERGE (c)-[:DEFINES {repo_id: $repo_id}]->(f)",
                    {"repo_id": config.repo_id, "class_id": class_id, "func_id": func_id},
                )

    # --- CALLS edges (deterministic) + VirtualCalls (unresolved dynamic) ---
    vc_seen: set[str] = set()

    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        module_name = module_names_by_path[stable_path]
        module_record = module_index[module_name]

        for func, cls in _iter_functions(module):
            src_qual = f"{cls.name}.{func.name}" if cls is not None else func.name
            src_id = _stable_function_id(stable_path, src_qual)

            # Deterministic edges from func.calls strings
            for call in sorted(func.calls):
                resolved = resolve_call_target(
                    call,
                    current_class=cls.name if cls is not None else None,
                    module_record=module_record,
                    import_maps=import_maps,
                    module_index=module_index,
                )

                if resolved.kind == "function" and resolved.module and resolved.qualname:
                    target_module = module_index.get(resolved.module)
                    if not target_module:
                        continue
                    dst_id = _stable_function_id(target_module.path, resolved.qualname)
                    mg.execute(
                        "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), (dst:Function {repo_id: $repo_id, id: $dst_id}) "
                        "MERGE (src)-[:CALLS {repo_id: $repo_id}]->(dst)",
                        {"repo_id": config.repo_id, "src_id": src_id, "dst_id": dst_id},
                    )
                else:
                    external_name = resolved.external_name or call
                    external_id = f"ext:{external_name}"
                    import_map_local = import_maps.get(module_record.name, {})
                    category = _classify_external(external_name, module_record=module_record, import_map=import_map_local)
                    mg.execute(
                        "MERGE (e:External {repo_id: $repo_id, id: $id}) "
                        "SET e.name = $name, e.namespace = $namespace, e.category = $category",
                        {
                            "repo_id": config.repo_id,
                            "id": external_id,
                            "name": external_name,
                            "namespace": _namespace_for_external(external_name),
                            "category": category,
                        },
                    )
                    mg.execute(
                        "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), (dst:External {repo_id: $repo_id, id: $dst_id}) "
                        "MERGE (src)-[:CALLS {repo_id: $repo_id}]->(dst)",
                        {"repo_id": config.repo_id, "src_id": src_id, "dst_id": external_id},
                    )

            # Virtual calls: only emit when it looks like dynamic dispatch AND not already resolved to internal
            for cd in getattr(func, "call_details", []):
                if not cd.callee_attr:
                    continue
                if cd.receiver_root_kind not in {"self", "cls", "super", "name"}:
                    continue

                # If the full_text was resolved to an internal Function above, no need for VirtualCall
                if cd.full_text:
                    r = resolve_call_target(
                        cd.full_text,
                        current_class=cls.name if cls is not None else None,
                        module_record=module_record,
                        import_maps=import_maps,
                        module_index=module_index,
                    )
                    if r.kind == "function":
                        continue

                vc_id = _virtual_call_id(src_id, cd.callee_attr, cd.receiver_root_name, cd.lineno)
                if vc_id not in vc_seen:
                    vc_seen.add(vc_id)
                    mg.execute(
                        "MERGE (v:VirtualCall {repo_id: $repo_id, id: $id}) "
                        "SET v.callee_attr = $callee_attr, v.receiver_kind = $receiver_kind, v.receiver_name = $receiver_name, "
                        "v.full_text = $full_text, v.lineno = $lineno, v.call_kind = $call_kind",
                        {
                            "repo_id": config.repo_id,
                            "id": vc_id,
                            "callee_attr": cd.callee_attr,
                            "receiver_kind": cd.receiver_root_kind,
                            "receiver_name": cd.receiver_root_name,
                            "full_text": cd.full_text,
                            "lineno": cd.lineno,
                            "call_kind": cd.call_kind,
                        },
                    )
                mg.execute(
                    "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), (v:VirtualCall {repo_id: $repo_id, id: $vc_id}) "
                    "MERGE (src)-[:CALLS_VIRTUAL {repo_id: $repo_id}]->(v)",
                    {"repo_id": config.repo_id, "src_id": src_id, "vc_id": vc_id},
                )

    # OVERRIDES edges
    _emit_overrides_edges(
        mg=mg,
        modules=modules,
        config=config,
        repo_root=repo_root,
        module_index=module_index,
        module_names_by_path=module_names_by_path,
        import_maps=import_maps,
    )

    # Side-effect signals (don’t clobber if already set)
    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        for func, cls in _iter_functions(module):
            src_qual = f"{cls.name}.{func.name}" if cls is not None else func.name
            src_id = _stable_function_id(stable_path, src_qual)

            for effect in infer_side_effects(func.calls):
                external_id = f"ext:{effect.call}"
                module_name = module_names_by_path[stable_path]
                module_record = module_index[module_name]
                import_map_local = import_maps.get(module_name, {})
                ext_category = _classify_external(effect.call, module_record=module_record, import_map=import_map_local)
                mg.execute(
                    "MERGE (e:External {repo_id: $repo_id, id: $id}) "
                    "SET e.name = $name, e.namespace = $namespace, e.category = $ext_category "
                    "WITH e "
                    "SET "
                    "e.side_effect_category = CASE WHEN e.side_effect_category IS NULL THEN $category ELSE e.side_effect_category END, "
                    "e.side_effect_confidence = CASE WHEN e.side_effect_confidence IS NULL THEN $confidence ELSE e.side_effect_confidence END, "
                    "e.side_effect_evidence = CASE WHEN e.side_effect_evidence IS NULL THEN $evidence ELSE e.side_effect_evidence END",
                    {
                        "repo_id": config.repo_id,
                        "id": external_id,
                        "name": effect.call,
                        "namespace": _namespace_for_external(effect.call),
                        "ext_category": ext_category,
                        "category": effect.category,
                        "confidence": effect.confidence,
                        "evidence": effect.evidence,
                    },
                )
                mg.execute(
                    "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), (dst:External {repo_id: $repo_id, id: $dst_id}) "
                    "MERGE (src)-[:CALLS {repo_id: $repo_id}]->(dst)",
                    {"repo_id": config.repo_id, "src_id": src_id, "dst_id": external_id},
                )
