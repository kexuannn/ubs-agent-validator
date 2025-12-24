from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import sys
import builtins

from gqlalchemy import Memgraph  # client for Memgraph graph DB

from algotracer.analysis.side_effects import infer_side_effects
from algotracer.ingest.ast_parser import ClassInfo, FunctionInfo, ModuleInfo, CallInfo


# ============================================================================
# Core data structures (module index, import targets, resolution results)
# ============================================================================


@dataclass(frozen=True)
class ModuleRecord:
    """Compact summary of one Python module used for resolution/lookups."""

    name: str                     # importable module name, e.g. "pkg.sub.mod"
    path: str                     # stable path (relative to repo_root)
    abs_path: str                 # absolute path on disk
    functions: set[str]           # top-level function names
    classes: set[str]             # class names
    class_methods: Dict[str, set[str]]  # class_name -> {method_name}
    class_bases: Dict[str, List[str]]   # class_name -> [base_name, ...]


@dataclass(frozen=True)
class ImportTarget:
    """How a local name (e.g. 'np', 'AlphaStrategy') maps to a module or symbol."""

    kind: str         # "module" or "symbol"
    module: str       # module path, e.g. "numpy", "vnpy.alpha"
    symbol: str | None  # symbol inside module if kind == "symbol"


@dataclass(frozen=True)
class ResolvedTarget:
    """Result of trying to resolve a call/base: internal function or external."""

    kind: str               # "function" or "external"
    module: str | None      # module that owns the function (if internal)
    qualname: str | None    # fully qualified name within the module
    external_name: str | None  # raw external name (if kind == "external")


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for one graph build run."""

    repo_id: str
    repo_root: Path


# ============================================================================
# Stable identifiers (modules, classes, functions)
# ============================================================================


def module_name_for_path(path: Path, repo_root: Path) -> str:
    """Convert a file path under repo_root into an import-like module name."""
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
    """Return a repo-relative posix path, or a hashed fallback if outside repo_root."""
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except Exception:
        h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
        return f"{path.name}.{h}"


def _stable_module_id(stable_path: str) -> str:
    """Canonical ID for a Module node."""
    return f"stable:mod:{stable_path}"


def _stable_function_id(stable_path: str, qualname: str) -> str:
    """Canonical ID for a Function node."""
    return f"stable:sym:{stable_path}:{qualname}"


def _stable_class_id(stable_path: str, qualname: str) -> str:
    """Canonical ID for a Class node."""
    return f"stable:cls:{stable_path}:{qualname}"


def build_module_records(modules: Sequence[ModuleInfo], repo_root: Path) -> Dict[str, ModuleRecord]:
    """
    Build a quick-lookup index of modules.

    Keyed by module_name ("pkg.sub.mod") → ModuleRecord summarising:
    - local functions
    - classes
    - methods per class
    - base classes per class
    """
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


# ============================================================================
# Import resolution helpers (relative imports, import maps, via-import calls)
# ============================================================================


def _resolve_relative_module(current_module: str, raw_module: str | None) -> str | None:
    """
    Turn a relative import module ('.utils', '..template') into an absolute
    module name based on the current module name.
    """
    if raw_module is None:
        return None
    if not raw_module.startswith("."):
        return raw_module

    # Count leading dots (relative depth), keep the suffix as the trailing module path.
    dots = len(raw_module) - len(raw_module.lstrip("."))
    suffix = raw_module.lstrip(".")
    base_parts = current_module.split(".") if current_module else []

    # Too many dots (going above package root) → give up and return suffix / None.
    if dots > len(base_parts):
        return suffix or None

    # Walk up `dots` levels, then optionally append the suffix.
    parent = base_parts[: len(base_parts) - dots]
    if suffix:
        parent.append(suffix)
    return ".".join([p for p in parent if p]) or None


def build_import_map(
    module: ModuleInfo,
    module_name: str,
    module_index: Dict[str, ModuleRecord],
) -> Dict[str, ImportTarget]:
    """
    Build a mapping for one module:
        local_name -> ImportTarget(kind="module"|"symbol", module=..., symbol=...)

    Examples:
        import numpy as np
            np -> module numpy

        from pkg import helpers
            helpers -> module pkg.helpers      (if pkg.helpers is a module)
            helpers -> symbol 'helpers' in pkg (otherwise)

        from .template import AlphaStrategy
            AlphaStrategy -> symbol 'AlphaStrategy' in pkg.template
    """
    mapping: Dict[str, ImportTarget] = {}

    for imp in module.imports:
        # Plain "import x" / "import x.y as z"
        if imp.kind == "import":
            if imp.alias:
                # import pkg.sub as ps
                mapping[imp.alias] = ImportTarget(kind="module", module=imp.name, symbol=None)
            else:
                # import pkg.sub  → we bind only the root "pkg"
                root = imp.name.split(".")[0]
                mapping[root] = ImportTarget(kind="module", module=root, symbol=None)
            continue

        # "from x import y" (possibly relative)
        resolved_module = _resolve_relative_module(module_name, imp.module)
        if not resolved_module:
            continue

        local_name = imp.alias or imp.name
        module_candidate = f"{resolved_module}.{imp.name}" if resolved_module else imp.name

        # If pkg.sub.y is itself a module, treat it as module import.
        if module_candidate in module_index:
            mapping[local_name] = ImportTarget(kind="module", module=module_candidate, symbol=None)
        else:
            # Otherwise treat it as a symbol inside the resolved_module.
            mapping[local_name] = ImportTarget(kind="symbol", module=resolved_module, symbol=imp.name)

    return mapping


def _resolve_via_imports(
    call: str,
    import_map: Dict[str, ImportTarget],
    module_index: Dict[str, ModuleRecord],
) -> ResolvedTarget | None:
    """
    Try to resolve a call string (e.g. "fn", "pkg.sub.fn", "pkg.Class.method")
    using import information only (no local context).
    """
    parts = call.split(".")
    if not parts:
        return None

    # Single-segment call, e.g. "helper_fn()"
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

    # Multi-segment: head must be a module imported in this file.
    head = parts[0]
    target = import_map.get(head)
    if not target or target.kind != "module":
        return None

    base_module = target.module
    remaining = parts[1:]

    # Try longest-prefix-first to find a module boundary:
    #   base_module + remaining[:i] as module, remaining[i:] as function/class tail.
    for i in range(len(remaining), 0, -1):
        candidate_module = base_module + "." + ".".join(remaining[:i])
        if candidate_module in module_index:
            mod = module_index[candidate_module]
            tail = remaining[i:]

            # pkg.sub.fn()
            if len(tail) == 1 and tail[0] in mod.functions:
                return ResolvedTarget(
                    kind="function",
                    module=candidate_module,
                    qualname=tail[0],
                    external_name=None,
                )

            # pkg.sub.Class.method()
            if (
                len(tail) == 2
                and tail[0] in mod.class_methods
                and tail[1] in mod.class_methods.get(tail[0], set())
            ):
                return ResolvedTarget(
                    kind="function",
                    module=candidate_module,
                    qualname=f"{tail[0]}.{tail[1]}",
                    external_name=None,
                )

    # Fallback: treat base_module as the real module and remaining as tail.
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

    if (
        len(remaining) == 2
        and remaining[0] in mod.class_methods
        and remaining[1] in mod.class_methods.get(remaining[0], set())
    ):
        return ResolvedTarget(
            kind="function",
            module=base_module,
            qualname=f"{remaining[0]}.{remaining[1]}",
            external_name=None,
        )

    return None


# ============================================================================
# Class / inheritance helpers (bases, re-exports, overrides)
# ============================================================================


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
    """
    Given a base class name and a method, try to find where base.method is defined.

    Handles:
    - local base (same module)
    - fully qualified base (pkg.mod.Class)
    - imported base symbol (from pkg import Class), including package re-exports.
    """
    # 1) Local class in the same module
    if base in module_record.class_methods and method in module_record.class_methods.get(base, set()):
        return ResolvedTarget(
            kind="function",
            module=module_record.name,
            qualname=f"{base}.{method}",
            external_name=None,
        )

    # 2) Fully qualified base name, e.g. "pkg.mod.Class"
    if "." in base:
        mod_name, class_name = base.rsplit(".", 1)
        mod = module_index.get(mod_name)
        if mod and method in mod.class_methods.get(class_name, set()):
            return ResolvedTarget(
                kind="function",
                module=mod.name,
                qualname=f"{class_name}.{method}",
                external_name=None,
            )

    # 3) Base imported as a symbol (possibly from a package)
    target = import_map.get(base)
    if target and target.kind == "symbol" and target.module:
        mod = module_index.get(target.module)
        class_name = target.symbol or base

        # 3a) Direct hit: the module is the defining module.
        if mod and method in mod.class_methods.get(class_name, set()):
            return ResolvedTarget(
                kind="function",
                module=mod.name,
                qualname=f"{class_name}.{method}",
                external_name=None,
            )

        # 3b) Fallback: class re-exported from deeper modules under this package.
        mod2, cls2 = _find_class_in_package(target.module, class_name, module_index)
        if mod2 and cls2 and method in mod2.class_methods.get(cls2, set()):
            return ResolvedTarget(
                kind="function",
                module=mod2.name,
                qualname=f"{cls2}.{method}",
                external_name=None,
            )

    return None


def _locate_base_class(
    base: str,
    *,
    module_record: ModuleRecord,
    module_index: Dict[str, ModuleRecord],
    import_map: Dict[str, ImportTarget],
) -> Tuple[ModuleRecord | None, str | None]:
    """
    Locate the ModuleRecord and class name for a base class string.

    Used for building SUBCLASS_OF edges (class inheritance graph).
    """
    # 1) Local class name
    if base in module_record.class_methods:
        return module_record, base

    # 2) Fully qualified "pkg.mod.Class"
    if "." in base:
        mod_name, cls_name = base.rsplit(".", 1)
        mod = module_index.get(mod_name)
        if mod and cls_name in mod.class_methods:
            return mod, cls_name

    # 3) Imported symbol (possibly re-exported from a package)
    target = import_map.get(base)
    if target and target.kind == "symbol" and target.module:
        mod = module_index.get(target.module)
        cls_name = target.symbol or base

        # Direct hit
        if mod and cls_name in mod.class_methods:
            return mod, cls_name

        # Fallback: scan modules under the package
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
    """
    Walk class hierarchy (including imported bases) to find where a method
    is originally defined.

    Used for:
    - resolving super().foo(...)
    - building OVERRIDES edges (child overrides parent method)
    """
    visited = visited or set()
    visited.add((module_record.name, current_class))

    for base in module_record.class_bases.get(current_class, []):
        # Try to resolve base.method at this level.
        candidate = _resolve_base_method(
            base,
            method,
            module_record=module_record,
            module_index=module_index,
            import_map=import_map,
        )
        if candidate is not None:
            return candidate

        # Otherwise, find the base class location and recurse into its bases.
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


# ============================================================================
# Call resolution within a module (local, bases, imported modules)
# ============================================================================


def _resolve_in_module(
    call: str,
    current_class: str | None,
    module_record: ModuleRecord,
    module_index: Dict[str, ModuleRecord],
    import_maps: Dict[str, Dict[str, ImportTarget]],
) -> ResolvedTarget | None:
    """
    Try to resolve a call string within the current module context:

    - self.foo / cls.foo               (methods on this class or bases)
    - super().foo                      (base class method)
    - unqualified foo() inside class   (method or base method)
    - Class.method                     (class method in same module)
    - top_level_func                   (module-level function)
    """
    import_map = import_maps.get(module_record.name, {})

    if current_class:
        parts = call.split(".")

        # super().foo(...) → resolve by walking base classes.
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

        # Direct self/cls method calls ONLY for `self.foo` / `cls.foo`.
        # For `self.x.foo` we do not treat it as a method on the current class
        # to avoid bogus self-loops like LassoModel.predict -> LassoModel.predict.
        if parts and parts[0] in {"self", "cls"}:
            if len(parts) == 2:
                method_name = parts[1]
                # self.foo / cls.foo on the same class
                if method_name in module_record.class_methods.get(current_class, set()):
                    return ResolvedTarget(
                        kind="function",
                        module=module_record.name,
                        qualname=f"{current_class}.{method_name}",
                        external_name=None,
                    )

                # Or inherited from bases
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
            # Deeper chains like `self.model.predict` fall through to External.

        # Unqualified method calls inside a class: `foo(...)`
        if len(parts) == 1 and call in module_record.class_methods.get(current_class, set()):
            return ResolvedTarget(
                kind="function",
                module=module_record.name,
                qualname=f"{current_class}.{call}",
                external_name=None,
            )

        # Unqualified method name that might live in a base class.
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

    # Class.method or module-level function in this module.
    parts = call.split(".")
    if len(parts) >= 2:
        class_name, method_name = parts[-2], parts[-1]
        if (
            class_name in module_record.class_methods
            and method_name in module_record.class_methods.get(class_name, set())
        ):
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
    """
    Unified entry point for resolving a call string:

    1. Try local/module + inheritance context (_resolve_in_module)
    2. Try imported modules and symbols (_resolve_via_imports)
    3. Otherwise classify as external
    """
    local = _resolve_in_module(call, current_class, module_record, module_index, import_maps)
    if local is not None:
        return local

    import_map = import_maps.get(module_record.name, {})
    via_imports = _resolve_via_imports(call, import_map, module_index)
    if via_imports is not None:
        return via_imports

    return ResolvedTarget(kind="external", module=None, qualname=None, external_name=call)


# ============================================================================
# Function/class iteration and VirtualCall identifiers
# ============================================================================


def _iter_functions(module: ModuleInfo) -> Iterable[tuple[FunctionInfo, ClassInfo | None]]:
    """Yield (function_or_method, owning_class_or_None) for all callables in a module."""
    for func in module.functions:
        yield func, None
    for cls in module.classes:
        for method in cls.methods:
            yield method, cls


def _namespace_for_external(name: str) -> str:
    """Namespace hint for External nodes (root segment before first dot)."""
    return name.split(".", 1)[0] if name else "unknown"


def _virtual_call_id(src_func_id: str, callee_attr: str | None, receiver: str | None, lineno: int | None) -> str:
    """Stable identifier for a VirtualCall node, based on source function + call site."""
    suffix = "|".join([callee_attr or "unknown", receiver or "unknown", str(lineno or 0)])
    h = hashlib.sha1(f"{src_func_id}:{suffix}".encode("utf-8")).hexdigest()[:12]
    return f"vc:{h}"


# ============================================================================
# External classification (stdlib / third_party / local_obj / unresolved)
# ============================================================================

# Precomputed sets for builtin and stdlib classification.
_BUILTIN_NAMES: set[str] = {name for name in dir(builtins) if not name.startswith("_")}
_STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", ())) or set(sys.builtin_module_names)


def _classify_external(
    external_name: str,
    *,
    module_record: ModuleRecord,
    import_map: Dict[str, ImportTarget],
) -> str:
    """Heuristic external classification.

    - Builtins like abs/len/int/float/list → "stdlib"
    - Names whose root matches a stdlib module (os, sys, pathlib, ...) → "stdlib"
    - Names whose root is imported in this module → "third_party"
    - Everything else → "unresolved"
    """
    root = (external_name or "").split(".", 1)[0]

    # handle call-roots like "set()" or "dict()"
    if root.endswith("()"):
        root = root[:-2]

    # 1) Builtin functions/types (abs, len, int, float, list, dict, set, etc.)
    if root in _BUILTIN_NAMES:
        return "stdlib"

    # 2) Stdlib modules (os, sys, json, pathlib, etc.)
    if root in _STDLIB_MODULES:
        return "stdlib"

    # 3) Anything imported in this module but not stdlib → treat as third_party
    if root in import_map:
        return "third_party"

    # 4) Default: unresolved
    return "unresolved"


def _classify_external_with_locals(
    external_name: str,
    *,
    module_record: ModuleRecord,
    import_map: Dict[str, ImportTarget],
    local_names: set[str],
) -> str:
    """
    Extended classification that first checks if the root is a local variable
    (e.g. `client`, `session`) and tags those as 'local_obj'.
    """
    root = (external_name or "").split(".", 1)[0]
    if root.endswith("()"):
        root = root[:-2]

    if root in local_names:
        return "local_obj"
    return _classify_external(
        external_name,
        module_record=module_record,
        import_map=import_map,
    )



def _vc_receiver_category(
    cd: CallInfo,
    *,
    module_record: ModuleRecord,
    import_map: Dict[str, ImportTarget],
    param_names: set[str],
    local_names: set[str],
) -> str:
    """
    Classify the receiver specifically for VirtualCall logic.

    Returns one of:
      - "this_obj"   : self / cls / super
      - "param_obj"  : function parameter object
      - "local_obj"  : local variable object
      - "stdlib"     : builtin / stdlib API
      - "third_party": imported lib object
      - "unresolved" : everything else
    """
    root_name = cd.receiver_root_name or ""
    root_kind = cd.receiver_root_kind

    # 1) Object bound to this instance
    if root_kind in {"self", "cls", "super"}:
        return "this_obj"

    # 2) Parameter / local variables
    if root_kind == "name":
        if root_name in param_names:
            return "param_obj"
        if root_name in local_names:
            return "local_obj"

    # 3) External fallback classification based on full_text/root
    external_name = cd.full_text or root_name
    cat = _classify_external(
        external_name,
        module_record=module_record,
        import_map=import_map,
    )
    return cat


# ============================================================================
# Receiver type inference (builtin / stdlib / third_party / user / unknown)
# ============================================================================


def _extract_self_field_name(cd: CallInfo) -> str | None:
    """
    For attribute calls on self/cls, extract the first field after self/cls.

    Example:
        self.pos_data.items -> "pos_data"
        cls.registry.get    -> "registry"
    """
    text = cd.full_text or ""
    parts = text.split(".")
    if len(parts) >= 2 and parts[0] in {"self", "cls"}:
        return parts[1]
    return None


def _normalize_type_root(type_expr: str) -> str:
    """
    Extract a root-ish token from a type expression.

    Examples:
        "dict[str, int]"     -> "dict"
        "list[Foo]"          -> "list"
        "httpx.Client"       -> "httpx.Client"
        "typing.Dict[str]"   -> "typing.Dict"
    """
    t = (type_expr or "").strip()
    if not t:
        return ""
    # strip generics / calls
    for sep in ("[", "("):
        if sep in t:
            t = t.split(sep, 1)[0]
            break
    return t.strip()


def _classify_type_expr(
    type_expr: str,
    *,
    module_record: ModuleRecord,
    import_map: Dict[str, ImportTarget],
    module_index: Dict[str, ModuleRecord],
) -> str:
    """
    Coarse classification for type expressions into:

      - builtin_container : dict/list/set/tuple
      - builtin_other     : int/float/str/bool/bytes/...
      - stdlib_class      : types coming from stdlib modules
      - third_party_class : imported non-stdlib modules
      - user_class        : symbols from modules in this repo
      - unknown           : we don't know
    """
    if not type_expr:
        return "unknown"

    root = _normalize_type_root(type_expr)
    if not root:
        return "unknown"

    # Split module vs attr: e.g. "httpx.Client" or "pkg.mod.Class"
    mod_part, _, _ = root.partition(".")
    head = mod_part or root

    # Builtin containers
    if head in {"dict", "list", "set", "tuple"}:
        return "builtin_container"

    # Simple builtin scalar-ish types
    if head in {"int", "float", "str", "bool", "bytes"}:
        return "builtin_other"

    # If the type name itself is a builtin (e.g. "list", "dict")
    if head in _BUILTIN_NAMES:
        # Most of these are not polymorphic in the way we care about
        return "builtin_other"

    # If the first segment is a stdlib module, treat as stdlib_class
    if head in _STDLIB_MODULES:
        return "stdlib_class"

    # If this symbol is imported, decide user vs third_party based on module_index
    target = import_map.get(head)
    if target:
        # Imported as module
        if target.kind == "module":
            if target.module in module_index:
                return "user_class"
            return "third_party_class"

        # Imported as symbol from a module
        if target.kind == "symbol":
            if target.module in module_index:
                return "user_class"
            return "third_party_class"

    # Maybe the type_expr itself is a module in our repo
    if root in module_index:
        return "user_class"

    # Default: unknown; we err on the side of keeping VCs
    return "unknown"


def _infer_self_field_type(
    *,
    module_record: ModuleRecord,
    class_name: str,
    field_name: str,
    module_index: Dict[str, ModuleRecord],
    import_maps: Dict[str, Dict[str, ImportTarget]],
    class_field_types: Dict[str, Dict[str, Dict[str, str]]],
    visited: set[tuple[str, str]] | None = None,
) -> str | None:
    """
    Walk this class and its bases to find a type expression for self.<field_name>.

    class_field_types is:
        module_name -> class_name -> {field_name -> type_expr}
    """
    visited = visited or set()
    key = (module_record.name, class_name)
    if key in visited:
        return None
    visited.add(key)

    # 1) Check this class
    module_types = class_field_types.get(module_record.name, {})
    field_map = module_types.get(class_name, {})
    t = field_map.get(field_name)
    if t:
        return t

    # 2) Walk bases
    import_map = import_maps.get(module_record.name, {})
    for base in module_record.class_bases.get(class_name, []):
        base_mod, base_name = _locate_base_class(
            base,
            module_record=module_record,
            module_index=module_index,
            import_map=import_map,
        )
        if not base_mod or not base_name:
            continue
        t2 = _infer_self_field_type(
            module_record=base_mod,
            class_name=base_name,
            field_name=field_name,
            module_index=module_index,
            import_maps=import_maps,
            class_field_types=class_field_types,
            visited=visited,
        )
        if t2:
            return t2

    return None


def _infer_receiver_type(
    cd: CallInfo,
    *,
    func: FunctionInfo,
    cls: ClassInfo | None,
    module_record: ModuleRecord,
    module_index: Dict[str, ModuleRecord],
    import_maps: Dict[str, Dict[str, ImportTarget]],
    class_field_types: Dict[str, Dict[str, Dict[str, str]]],
) -> str:
    """
    Infer a coarse receiver type for VC gating:

      - builtin_container / builtin_other
      - stdlib_class / third_party_class
      - user_class
      - unknown
    """
    root_kind = cd.receiver_root_kind
    root_name = cd.receiver_root_name or ""
    import_map = import_maps.get(module_record.name, {})

    # 1) self/cls fields: self.pos_data.items(...)
    if root_kind in {"self", "cls"} and cls is not None:
        field_name = _extract_self_field_name(cd)
        if field_name:
            t_expr = _infer_self_field_type(
                module_record=module_record,
                class_name=cls.name,
                field_name=field_name,
                module_index=module_index,
                import_maps=import_maps,
                class_field_types=class_field_types,
            )
            if t_expr:
                return _classify_type_expr(
                    t_expr,
                    module_record=module_record,
                    import_map=import_map,
                    module_index=module_index,
                )
        # Fallback: we don't know field type; treat self.* as user_class-ish
        return "user_class"

    # 2) Plain name: param or local (duck-typing candidate, or alias of self)
    if root_kind == "name":
        # 2a) alias of a self/cls field:
        #     target = self.chart_item
        #     target.update(...)
        alias_map: dict[str, str] = getattr(func, "self_aliases", {}) or {}
        if cls is not None and root_name in alias_map:
            field_name = alias_map[root_name]
            t_expr = _infer_self_field_type(
                module_record=module_record,
                class_name=cls.name,
                field_name=field_name,
                module_index=module_index,
                import_maps=import_maps,
                class_field_types=class_field_types,
            )
            if t_expr:
                return _classify_type_expr(
                    t_expr,
                    module_record=module_record,
                    import_map=import_map,
                    module_index=module_index,
                )
            # Even if we don't know the exact type, alias-of-self is still "user-ish"
            return "user_class"

        # 2b) Normal param/local with annotations/inferred type
        t_expr = getattr(func, "param_type_hints", {}).get(root_name)
        if not t_expr:
            t_expr = getattr(func, "local_type_hints", {}).get(root_name)
        if not t_expr:
            t_expr = getattr(func, "local_type_inferred", {}).get(root_name)

        if t_expr:
            return _classify_type_expr(
                t_expr,
                module_record=module_record,
                import_map=import_map,
                module_index=module_index,
            )

        # Unknown param/local type → keep as unknown (we want VCs for duck-typed params)
        return "unknown"

    # 3) Other roots (call/subscript/unknown)
    external_name = cd.full_text or root_name
    cat = _classify_external(
        external_name,
        module_record=module_record,
        import_map=import_map,
    )

    if cat == "stdlib":
        return "stdlib_class"
    if cat == "third_party":
        return "third_party_class"

    # unresolved → unknown; err on side of keeping VC
    return "unknown"


# ============================================================================
# Overrides (OVERRIDES edges between overriding and base methods)
# ============================================================================


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
    """
    For each method on each class, find the base method it overrides and emit
    an OVERRIDES edge: child_method -[:OVERRIDES]-> base_method.
    """
    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        module_name = module_names_by_path[stable_path]
        module_record = module_index[module_name]
        import_map = import_maps.get(module_record.name, {})

        for cls in module.classes:
            # Skip classes with no bases.
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


# ============================================================================
# Main graph write pipeline (modules, classes, functions, calls, side-effects)
# ============================================================================


def write_graph(*, mg: Memgraph, modules: Sequence[ModuleInfo], config: GraphBuildConfig) -> None:
    """
    Ingest a set of parsed modules into Memgraph:

    1. Build module index + import maps
    2. Create Repo / Module / Class / Function nodes and structural edges
    3. Emit CALLS edges (internal + External nodes)
    4. Emit VirtualCall nodes + CALLS_VIRTUAL edges (for polymorphic-ish sites)
    5. Emit OVERRIDES edges (inheritance-aware)
    6. Annotate External nodes with side-effect signals
    """
    repo_root = config.repo_root
    module_index = build_module_records(modules, repo_root)

    module_names_by_path = {
        _stable_path(m.path, repo_root): module_name_for_path(m.path, repo_root)
        for m in modules
    }

    # Precompute import maps for each module name.
    import_maps: Dict[str, Dict[str, ImportTarget]] = {}
    for module in modules:
        name = module_name_for_path(module.path, repo_root)
        import_maps[name] = build_import_map(module, name, module_index)

    # Precompute class field type maps:
    #   module_name -> class_name -> {field_name -> type_expr}
    class_field_types: Dict[str, Dict[str, Dict[str, str]]] = {}
    for module in modules:
        module_name = module_name_for_path(module.path, repo_root)
        per_class: Dict[str, Dict[str, str]] = {}
        for cls in module.classes:
            per_class[cls.name] = dict(getattr(cls, "field_types", {}) or {})
        class_field_types[module_name] = per_class

    # Repo node
    mg.execute(
        "MERGE (r:Repo {repo_id: $repo_id}) SET r.root = $root",
        {"repo_id": config.repo_id, "root": str(repo_root.resolve())},
    )

    # -----------------------------------------------------------------------
    # Modules, Classes, Functions (structure)
    # -----------------------------------------------------------------------
    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        module_name = module_names_by_path[stable_path]
        module_id = _stable_module_id(stable_path)

        # Module node
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

        # Class nodes and Module->Class edges
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

        # Class inheritance edges (SUBCLASS_OF)
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
                    "MATCH (child:Class {repo_id: $repo_id, id: $child_id}), "
                    "(parent:Class {repo_id: $repo_id, id: $parent_id}) "
                    "MERGE (child)-[:SUBCLASS_OF {repo_id: $repo_id}]->(parent)",
                    {"repo_id": config.repo_id, "child_id": child_id, "parent_id": parent_id},
                )

        # Function nodes and structural edges (Module->Function, Class->Function)
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

    # -----------------------------------------------------------------------
    # Deterministic CALLS + VirtualCall nodes (dynamic dispatch sites)
    # -----------------------------------------------------------------------
    vc_seen: set[str] = set()

    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        module_name = module_names_by_path[stable_path]
        module_record = module_index[module_name]
        import_map_local = import_maps.get(module_record.name, {})

        for func, cls in _iter_functions(module):
            src_qual = f"{cls.name}.{func.name}" if cls is not None else func.name
            src_id = _stable_function_id(stable_path, src_qual)

            param_names = set(getattr(func, "params", set()))
            local_names = set(getattr(func, "assigned_locals", set()))
            all_localish = param_names | local_names

            # --- Deterministic CALLS edges from func.calls strings ---
            for call in sorted(func.calls):
                resolved = resolve_call_target(
                    call,
                    current_class=cls.name if cls is not None else None,
                    module_record=module_record,
                    import_maps=import_maps,
                    module_index=module_index,
                )

                if resolved.kind == "function" and resolved.module and resolved.qualname:
                    # Internal call: Function -> Function
                    target_module = module_index.get(resolved.module)
                    if not target_module:
                        continue
                    dst_id = _stable_function_id(target_module.path, resolved.qualname)
                    mg.execute(
                        "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), "
                        "(dst:Function {repo_id: $repo_id, id: $dst_id}) "
                        "MERGE (src)-[:CALLS {repo_id: $repo_id}]->(dst)",
                        {"repo_id": config.repo_id, "src_id": src_id, "dst_id": dst_id},
                    )
                else:
                    # External call: Function -> External
                    external_name = resolved.external_name or call
                    external_id = f"ext:{external_name}"
                    category = _classify_external_with_locals(
                        external_name,
                        module_record=module_record,
                        import_map=import_map_local,
                        local_names=all_localish,
                    )
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
                        "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), "
                        "(dst:External {repo_id: $repo_id, id: $dst_id}) "
                        "MERGE (src)-[:CALLS {repo_id: $repo_id}]->(dst)",
                        {"repo_id": config.repo_id, "src_id": src_id, "dst_id": external_id},
                    )

            # --- Virtual calls: dynamic OO-ish / duck-typed dispatch sites ---
            for cd in getattr(func, "call_details", []):
                # must be attribute-style: receiver.attr(...)
                if not cd.callee_attr or cd.call_kind != "attr_call":
                    continue

                # try deterministic resolution first
                if cd.full_text:
                    r = resolve_call_target(
                        cd.full_text,
                        current_class=cls.name if cls is not None else None,
                        module_record=module_record,
                        import_maps=import_maps,
                        module_index=module_index,
                    )
                    if r.kind == "function":
                        # Already has a precise CALLS edge
                        continue

                receiver_cat = _vc_receiver_category(
                    cd,
                    module_record=module_record,
                    import_map=import_map_local,
                    param_names=param_names,
                    local_names=local_names,
                )

                # Infer coarse receiver type using param/local/self field types
                receiver_type = _infer_receiver_type(
                    cd,
                    func=func,
                    cls=cls,
                    module_record=module_record,
                    module_index=module_index,
                    import_maps=import_maps,
                    class_field_types=class_field_types,
                )

                # Strong negative signals: clearly non-polymorphic / library-ish types
                if receiver_type in {
                    "builtin_container",
                    "builtin_other",
                    "stdlib_class",
                    "third_party_class",
                }:
                    # e.g. self.pos_data.items(), cache.update(), httpx.Client().get()
                    # → treat as boring library/container, no VC
                    continue

                # Only meaningful VC receivers based on root-kind:
                # - this_obj   (self/cls/super)
                # - local_obj
                # - param_obj  (dependency objects / duck-typed params)
                # - unresolved (unknown but still interesting)
                if receiver_cat not in {"this_obj", "param_obj", "unresolved", "local_obj"}:
                    continue

                vc_id = _virtual_call_id(
                    src_id, cd.callee_attr, cd.receiver_root_name, cd.lineno
                )

                if vc_id not in vc_seen:
                    vc_seen.add(vc_id)
                    mg.execute(
                        "MERGE (v:VirtualCall {repo_id: $repo_id, id: $id}) "
                        "SET v.callee_attr = $callee_attr, v.receiver_kind = $receiver_kind, "
                        "v.receiver_name = $receiver_name, v.full_text = $full_text, "
                        "v.lineno = $lineno, v.call_kind = $call_kind",
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
                    "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), "
                    "(v:VirtualCall {repo_id: $repo_id, id: $vc_id}) "
                    "MERGE (src)-[:CALLS_VIRTUAL {repo_id: $repo_id}]->(v)",
                    {"repo_id": config.repo_id, "src_id": src_id, "vc_id": vc_id},
                )

    # -----------------------------------------------------------------------
    # OVERRIDES edges (method override relationships)
    # -----------------------------------------------------------------------
    _emit_overrides_edges(
        mg=mg,
        modules=modules,
        config=config,
        repo_root=repo_root,
        module_index=module_index,
        module_names_by_path=module_names_by_path,
        import_maps=import_maps,
    )

    # -----------------------------------------------------------------------
    # Side-effect signals on External nodes
    # -----------------------------------------------------------------------
    for module in modules:
        stable_path = _stable_path(module.path, repo_root)
        for func, cls in _iter_functions(module):
            src_qual = f"{cls.name}.{func.name}" if cls is not None else func.name
            src_id = _stable_function_id(stable_path, src_qual)
            local_names = set(getattr(func, "params", set())) | set(
                getattr(func, "assigned_locals", set())
            )

            for effect in infer_side_effects(func.calls):
                external_id = f"ext:{effect.call}"
                module_name = module_names_by_path[stable_path]
                module_record = module_index[module_name]
                import_map_local = import_maps.get(module_name, {})
                ext_category = _classify_external_with_locals(
                    effect.call,
                    module_record=module_record,
                    import_map=import_map_local,
                    local_names=local_names,
                )
                mg.execute(
                    "MERGE (e:External {repo_id: $repo_id, id: $id}) "
                    "SET e.name = $name, e.namespace = $namespace, e.category = $ext_category "
                    "WITH e "
                    "SET "
                    "e.side_effect_category = CASE WHEN e.side_effect_category IS NULL "
                    "THEN $category ELSE e.side_effect_category END, "
                    "e.side_effect_confidence = CASE WHEN e.side_effect_confidence IS NULL "
                    "THEN $confidence ELSE e.side_effect_confidence END, "
                    "e.side_effect_evidence = CASE WHEN e.side_effect_evidence IS NULL "
                    "THEN $evidence ELSE e.side_effect_evidence END",
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
                    "MATCH (src:Function {repo_id: $repo_id, id: $src_id}), "
                    "(dst:External {repo_id: $repo_id, id: $dst_id}) "
                    "MERGE (src)-[:CALLS {repo_id: $repo_id}]->(dst)",
                    {"repo_id": config.repo_id, "src_id": src_id, "dst_id": external_id},
                )
