from __future__ import annotations

"""AST parsing helpers to extract modules, imports, classes, functions, and call relations.

Instructions:
- Use parse_module(path) to parse a single Python file into ModuleInfo.
- Use parse_python_files(iterable_of_paths) to parse many files at once.
- Extend _SymbolVisitor if you need additional node data (e.g., assignments or attributes).

Explanation:
- _SymbolVisitor walks the AST to collect imports, class definitions, and function definitions.
- _collect_calls gathers called function names within each function/method body.
- Decorators are captured as dotted names (including attribute chains) for use in heuristics.
- ModuleInfo aggregates the collected symbols for later graph building.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set


@dataclass
class FunctionInfo:
    name: str
    lineno: int
    calls: Set[str] = field(default_factory=set)
    decorators: List[str] = field(default_factory=list)  # dotted decorator names


@dataclass
class ClassInfo:
    name: str
    lineno: int
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    path: Path
    imports: List[str] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: List[str] = []
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []
        self._class_stack: List[ClassInfo] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        for alias in node.names:
            target = f"{module}.{alias.name}" if module else alias.name
            self.imports.append(target)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        decos: List[str] = []
        for d in node.decorator_list:
            name = _decorator_name(d)
            if name:
                decos.append(name)

        func = FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            decorators=decos,
        )
        func.calls = _collect_calls(node)

        if self._class_stack:
            self._class_stack[-1].methods.append(func)
        else:
            self.functions.append(func)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        cls = ClassInfo(
            name=node.name,
            lineno=node.lineno,
            bases=[_base_name(base) for base in node.bases],
        )
        self._class_stack.append(cls)
        self.generic_visit(node)
        self._class_stack.pop()
        self.classes.append(cls)


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _call_name(node.func)
        if name:
            self.calls.add(name)
        self.generic_visit(node)


def _collect_calls(node: ast.AST) -> Set[str]:
    collector = _CallCollector()
    collector.visit(node)
    return collector.calls


def _call_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Attribute):
        root = _call_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ast.unparse(node)


def _decorator_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        root = _decorator_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def parse_module(path: Path) -> ModuleInfo:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _SymbolVisitor()
    visitor.visit(tree)
    return ModuleInfo(path=path, imports=visitor.imports, classes=visitor.classes, functions=visitor.functions)


def parse_python_files(paths: Iterable[Path]) -> List[ModuleInfo]:
    return [parse_module(path) for path in paths]
