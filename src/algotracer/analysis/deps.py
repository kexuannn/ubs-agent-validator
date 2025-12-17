from __future__ import annotations

"""Dependency graph construction utilities.

Typed node IDs:

- mod:<path>                 = a Python file/module node
- sym:<path>:<qualname>      = a function/method node defined in that file
- ext:<name>                 = an external/unresolved target (import or call)

Wiring:
- mod -> ext(imports)
- mod -> sym(symbols defined in the module)
- sym -> sym (best-effort, intra-module resolution)
- sym -> ext (unresolved calls)

Reverse wiring:
- callers(node) lets you ask "who points to me?" for any node.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

from algotracer.ingest.ast_parser import ModuleInfo


@dataclass
class DependencyGraph:
    edges: Dict[str, Set[str]] = field(default_factory=dict)
    reverse_edges: Dict[str, Set[str]] = field(default_factory=dict)
    nodes: Set[str] = field(default_factory=set)

    def add_node(self, node: str) -> None:
        self.nodes.add(node)

    def add_edge(self, src: str, dest: str) -> None:
        self.add_node(src)
        self.add_node(dest)
        self.edges.setdefault(src, set()).add(dest)
        self.reverse_edges.setdefault(dest, set()).add(src)

    def neighbors(self, node: str) -> Set[str]:
        return self.edges.get(node, set())

    def callers(self, node: str) -> Set[str]:
        """Reverse lookup: who points to this node?"""
        return self.reverse_edges.get(node, set())


def _mod_node(module_path: str) -> str:
    return f"mod:{module_path}"


def _sym_node(module_path: str, qualname: str) -> str:
    return f"sym:{module_path}:{qualname}"


def _ext_node(name: str) -> str:
    return f"ext:{name}"


def summarize_callers(graph: DependencyGraph, node: str, limit: int = 10) -> list[str]:
    """Return up to `limit` callers of a node, sorted for determinism."""
    return sorted(graph.callers(node))[:limit]


def _resolve_call_in_module(
    *,
    module_path: str,
    call: str,
    current_class: str | None,
    top_level_funcs: Set[str],
    class_methods: Dict[str, Set[str]],
) -> str:
    """Best-effort resolution for calls to symbols defined in the SAME module.

    Resolution order (lightweight, static):
    - self.foo / cls.foo / bare foo    -> current class method
    - ClassName.foo                    -> class method in same module
    - foo                              -> top-level function in module
    Otherwise returns ext:<call>.
    """
    # Case 1: current class method (self.foo / cls.foo / bare foo)
    if current_class:
        parts = call.split(".")
        if parts[0] in {"self", "cls"} and len(parts) >= 2:
            method_name = parts[1]
            if method_name in class_methods.get(current_class, set()):
                return _sym_node(module_path, f"{current_class}.{method_name}")
        if len(parts) == 1 and call in class_methods.get(current_class, set()):
            return _sym_node(module_path, f"{current_class}.{call}")

    # Case 2: explicit ClassName.method (only if it looks like a class or is known)
    parts = call.split(".")
    if len(parts) >= 2:
        class_name, method_name = parts[-2], parts[-1]
        if (class_name in class_methods or class_name[:1].isupper()) and method_name in class_methods.get(class_name, set()):
            return _sym_node(module_path, f"{class_name}.{method_name}")

    # Case 3: bare function name
    if call in top_level_funcs:
        return _sym_node(module_path, call)

    # Otherwise unresolved (could be imported, builtin, external lib, etc.)
    return _ext_node(call)


def build_dependency_graph(modules: Iterable[ModuleInfo]) -> DependencyGraph:
    graph = DependencyGraph()

    for module in modules:
        module_path = str(module.path)
        mod = _mod_node(module_path)
        graph.add_node(mod)

        # Build a quick "symbol table" for intra-module resolution
        top_level_funcs: Set[str] = {f.name for f in module.functions}
        class_methods: Dict[str, Set[str]] = {
            cls.name: {m.name for m in cls.methods} for cls in module.classes
        }

        # 1) Module -> Imports (still external targets for now)
        for imp in module.imports:
            graph.add_edge(mod, _ext_node(imp))

        # 2) Module -> Symbols, Symbol -> Calls (resolved to sym where possible)

        # Classes + methods
        for cls in module.classes:
            for method in cls.methods:
                qualname = f"{cls.name}.{method.name}"
                sym = _sym_node(module_path, qualname)

                graph.add_edge(mod, sym)  # module contains this method

                for call in method.calls:
                    dest = _resolve_call_in_module(
                        module_path=module_path,
                        call=call,
                        current_class=cls.name,
                        top_level_funcs=top_level_funcs,
                        class_methods=class_methods,
                    )
                    graph.add_edge(sym, dest)

        # Top-level functions
        for func in module.functions:
            sym = _sym_node(module_path, func.name)
            graph.add_edge(mod, sym)  # module contains this function

            for call in func.calls:
                dest = _resolve_call_in_module(
                    module_path=module_path,
                    call=call,
                    current_class=None,
                    top_level_funcs=top_level_funcs,
                    class_methods=class_methods,
                )
                graph.add_edge(sym, dest)

    return graph
