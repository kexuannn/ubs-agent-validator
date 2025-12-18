from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from gqlalchemy import Memgraph

from algotracer.ingest.ast_parser import ClassInfo, FunctionInfo, ModuleInfo


@dataclass(frozen=True)
class ModuleRecord:
    name: str
    path: str
    abs_path: str
    functions: set[str]
    classes: set[str]
    class_methods: Dict[str, set[str]]


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
        rel = path.name  # fallback to basename if outside repo_root
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
        return path.name


def _stable_module_id(stable_path: str) -> str:
    return f"stable:mod:{stable_path}"


def _stable_function_id(stable_path: str, qualname: str) -> str:
    return f"stable:sym:{stable_path}:{qualname}"


def build_module_records(modules: Sequence[ModuleInfo], repo_root: Path) -> Dict[str, ModuleRecord]:
    records: Dict[str, ModuleRecord] = {}
    for module in modules:
        name = module_name_for_path(module.path, repo_root)
        stable_path = _stable_path(module.path, repo_root)
        functions = {f.name for f in module.functions}
        classes = {c.name for c in module.classes}
        class_methods = {c.name: {m.name for m in c.methods} for c in module.classes}
        records[name] = ModuleRecord(
            name=name,
            path=stable_path,
            abs_path=str(module.path.resolve()),
            functions=functions,
            classes=classes,
            class_methods=class_methods,
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
                mapping[imp.alias] = ImportTarget(
                    kind="module",
                    module=imp.name,
                    symbol=None,
                )
            else:
                root = imp.name.split(".")[0]
                mapping[root] = ImportTarget(
                    kind="module",
                    module=root,
                    symbol=None,
                )
            continue

        resolved_module = _resolve_relative_module(module_name, imp.module)
        if not resolved_module:
            continue

        local_name = imp.alias or imp.name
        module_candidate = f"{resolved_module}.{imp.name}" if resolved_module else imp.name
        if module_candidate in module_index:
            mapping[local_name] = ImportTarget(
                kind="module",
                module=module_candidate,
                symbol=None,
            )
        else:
            mapping[local_name] = ImportTarget(
                kind="symbol",
                module=resolved_module,
                symbol=imp.name,
            )

    return mapping


def _resolve_in_module(
    call: str,
    current_class: str | None,
    module_record: ModuleRecord,
) -> ResolvedTarget | None:
    if current_class:
        parts = call.split(".")
        if parts[0] in {"self", "cls"} and len(parts) >= 2:
            method_name = parts[1]
            if method_name in module_record.class_methods.get(current_class, set()):
                return ResolvedTarget(
                    kind="function",
                    module=module_record.name,
                    qualname=f"{current_class}.{method_name}",
                    external_name=None,
                )
        if len(parts) == 1 and call in module_record.class_methods.get(current_class, set()):
            return ResolvedTarget(
                kind="function",
                module=module_record.name,
                qualname=f"{current_class}.{call}",
                external_name=None,
            )

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
                return ResolvedTarget(
                    kind="function",
                    module=target.module,
                    qualname=target.symbol,
                    external_name=None,
                )
        return None

    head = parts[0]
    target = import_map.get(head)
    if not target or target.kind != "module":
        return None

    base_module = target.module
    remaining = parts[1:]

    # Try to find a deeper module match (conservative)
    for i in range(len(remaining), 0, -1):
        candidate_module = base_module + "." + ".".join(remaining[:i])
        if candidate_module in module_index:
            mod = module_index[candidate_module]
            tail = remaining[i:]
            if len(tail) == 1 and tail[0] in mod.functions:
                return ResolvedTarget(
                    kind="function",
                    module=candidate_module,
                    qualname=tail[0],
                    external_name=None,
                )
            if len(tail) == 2 and tail[0] in mod.class_methods and tail[1] in mod.class_methods.get(tail[0], set()):
                return ResolvedTarget(
                    kind="function",
                    module=candidate_module,
                    qualname=f"{tail[0]}.{tail[1]}",
                    external_name=None,
                )

    # Resolve against base module as Class.method or function
    mod = module_index.get(base_module)
    if not mod:
        return None
    if len(remaining) == 1 and remaining[0] in mod.functions:
        return ResolvedTarget(
            kind="function",
            module=base_module,
            qualname=remaining[0],
            external_name=None,
        )
    if len(remaining) == 2 and remaining[0] in mod.class_methods and remaining[1] in mod.class_methods.get(remaining[0], set()):
        return ResolvedTarget(
            kind="function",
            module=base_module,
            qualname=f"{remaining[0]}.{remaining[1]}",
            external_name=None,
        )

    return None


def resolve_call_target(
    call: str,
    *,
    current_class: str | None,
    module_record: ModuleRecord,
    import_map: Dict[str, ImportTarget],
    module_index: Dict[str, ModuleRecord],
) -> ResolvedTarget:
    local = _resolve_in_module(call, current_class, module_record)
    if local is not None:
        return local

    via_imports = _resolve_via_imports(call, import_map, module_index)
    if via_imports is not None:
        return via_imports

    return ResolvedTarget(
        kind="external",
        module=None,
        qualname=None,
        external_name=call,
    )


def _iter_functions(module: ModuleInfo) -> Iterable[tuple[FunctionInfo, ClassInfo | None]]:
    for func in module.functions:
        yield func, None
    for cls in module.classes:
        for method in cls.methods:
            yield method, cls


def _namespace_for_external(name: str) -> str:
    return name.split(".", 1)[0] if name else "unknown"


def write_graph(
    *,
    mg: Memgraph,
    modules: Sequence[ModuleInfo],
    config: GraphBuildConfig,
) -> None:
    repo_root = config.repo_root
    module_index = build_module_records(modules, repo_root)

    module_names_by_path = {
        _stable_path(m.path, repo_root): module_name_for_path(m.path, repo_root)
        for m in modules
    }

    import_maps: Dict[str, Dict[str, ImportTarget]] = {}
    for module in modules:
        name = module_name_for_path(module.path, repo_root)
        import_maps[name] = build_import_map(module, name, module_index)

    # Repo node
    mg.execute(
        "MERGE (r:Repo {repo_id: $repo_id}) SET r.root = $root",
        {"repo_id": config.repo_id, "root": str(repo_root.resolve())},
    )

    # Create module and function nodes
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

        for func, cls in _iter_functions(module):
            qualname = f"{cls.name}.{func.name}" if cls is not None else func.name
            func_id = _stable_function_id(stable_path, qualname)
            kind = "method" if cls is not None else "function"

            mg.execute(
                "MERGE (f:Function {repo_id: $repo_id, id: $id}) "
                "SET f.name = $name, f.qualname = $qualname, f.path = $path, "
                "f.lineno = $lineno, f.kind = $kind, f.signature = $signature",
                {
                    "repo_id": config.repo_id,
                    "id": func_id,
                    "name": func.name,
                    "qualname": qualname,
                    "path": stable_path,
                    "lineno": func.lineno,
                    "kind": kind,
                    "signature": func.signature,
                },
            )

            mg.execute(
                "MATCH (m:Module {repo_id: $repo_id, id: $module_id}), "
                "(f:Function {repo_id: $repo_id, id: $func_id}) "
                "MERGE (m)-[:DEFINES {repo_id: $repo_id}]->(f)",
                {
                    "repo_id": config.repo_id,
                    "module_id": module_id,
                    "func_id": func_id,
                },
            )

    # Create CALLS edges
    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        module_name = module_names_by_path[stable_path]
        module_record = module_index[module_name]
        import_map = import_maps[module_name]

        for func, cls in _iter_functions(module):
            qualname = f"{cls.name}.{func.name}" if cls is not None else func.name
            src_id = _stable_function_id(stable_path, qualname)

            for call in sorted(func.calls):
                resolved = resolve_call_target(
                    call,
                    current_class=cls.name if cls is not None else None,
                    module_record=module_record,
                    import_map=import_map,
                    module_index=module_index,
                )

                if resolved.kind == "function" and resolved.module and resolved.qualname:
                    target_module = module_index.get(resolved.module)
                    if not target_module:
                        continue
                    dest_id = _stable_function_id(target_module.path, resolved.qualname)
                    mg.execute(
                        "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), "
                        "(dst:Function {repo_id: $repo_id, id: $dst_id}) "
                        "MERGE (src)-[:CALLS {repo_id: $repo_id}]->(dst)",
                        {
                            "repo_id": config.repo_id,
                            "src_id": src_id,
                            "dst_id": dest_id,
                        },
                    )
                else:
                    external_name = resolved.external_name or call
                    external_id = f"ext:{external_name}"
                    mg.execute(
                        "MERGE (e:External {repo_id: $repo_id, id: $id}) "
                        "SET e.name = $name, e.namespace = $namespace",
                        {
                            "repo_id": config.repo_id,
                            "id": external_id,
                            "name": external_name,
                            "namespace": _namespace_for_external(external_name),
                        },
                    )
                    mg.execute(
                        "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), "
                        "(dst:External {repo_id: $repo_id, id: $dst_id}) "
                        "MERGE (src)-[:CALLS {repo_id: $repo_id}]->(dst)",
                        {
                            "repo_id": config.repo_id,
                            "src_id": src_id,
                            "dst_id": external_id,
                        },
                    )
