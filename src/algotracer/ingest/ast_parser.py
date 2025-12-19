from __future__ import annotations

"""
AST parsing helpers to extract modules, imports, classes, functions, and call relations.

Key upgrades:
- Collects BOTH:
  (a) function.calls: Set[str] of best-effort normalized call strings (backward compatible)
  (b) function.call_details: List[CallInfo] with receiver/callee metadata for robust virtual resolution
- Fixes receiver_root_kind classification:
  - self / cls / super are detected correctly
  - attribute-chains keep the true root kind (e.g. self.x.foo -> self)
- Handles ast.Call and ast.Subscript roots in call name normalization.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set


@dataclass(frozen=True)
class ImportInfo:
    module: str | None
    name: str
    alias: str | None
    kind: str  # "import" or "from"


@dataclass(frozen=True)
class CallInfo:
    callee_attr: str | None
    receiver_root_kind: str  # self|cls|super|name|call|subscript|unknown
    receiver_root_name: str | None
    full_text: str | None
    call_kind: str  # attr_call | name_call
    lineno: int | None
    col: int | None
    current_class: str | None


@dataclass
class FunctionInfo:
    name: str
    lineno: int
    signature: str | None = None
    calls: Set[str] = field(default_factory=set)
    call_details: List[CallInfo] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    lineno: int
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    path: Path
    imports: List[ImportInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: List[ImportInfo] = []
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []
        self._class_stack: List[ClassInfo] = []

    def _current_class_name(self) -> str | None:
        return self._class_stack[-1].name if self._class_stack else None

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=None,
                    name=alias.name,
                    alias=alias.asname,
                    kind="import",
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = f"{'.' * node.level}{node.module or ''}"
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=module or None,
                    name=alias.name,
                    alias=alias.asname,
                    kind="from",
                )
            )
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
            signature=_format_signature(node),
            decorators=decos,
        )

        calls, details = _collect_calls(node, current_class=self._current_class_name())
        func.calls = calls
        func.call_details = details

        if self._class_stack:
            self._class_stack[-1].methods.append(func)
        else:
            self.functions.append(func)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.visit_FunctionDef(node)

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
    def __init__(self, current_class: str | None) -> None:
        self.calls: Set[str] = set()
        self.call_details: List[CallInfo] = []
        self._current_class = current_class

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        full = _call_name(node.func)
        root_kind, root_name, callee_attr, call_kind = _call_metadata(node.func)

        if full:
            self.calls.add(full)

        self.call_details.append(
            CallInfo(
                callee_attr=callee_attr,
                receiver_root_kind=root_kind,
                receiver_root_name=root_name,
                full_text=full,
                call_kind=call_kind,
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                current_class=self._current_class,
            )
        )
        self.generic_visit(node)


def _collect_calls(node: ast.AST, current_class: str | None) -> tuple[Set[str], List[CallInfo]]:
    collector = _CallCollector(current_class)
    collector.visit(node)
    return collector.calls, collector.call_details


def _call_name(node: ast.AST | None) -> str | None:
    """
    Best-effort normalized call name extraction.

    Examples:
      foo()                -> "foo"
      self.foo()           -> "self.foo"
      self.x.foo()         -> "self.x.foo"
      super().foo()        -> "super().foo"
      self.builder().foo() -> "self.builder().foo" (keeps self; we don't drop root)
      items[0].foo()       -> "items[].foo"
      x().y().z()          -> "x().y().z"
    """
    if node is None:
        return None

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        root = _call_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr

    if isinstance(node, ast.Call):
        fn = _call_name(node.func)
        if fn == "super":
            return "super()"
        return f"{fn}()" if fn else None

    if isinstance(node, ast.Subscript):
        root = _call_name(node.value)
        return f"{root}[]" if root else None

    return None


def _call_metadata(node: ast.AST | None) -> tuple[str, str | None, str | None, str]:
    """
    Return (receiver_root_kind, receiver_root_name, callee_attr, call_kind).

    Crucial behavior:
    - For attribute chains, receiver_root_kind/name reflect the *true root* (self/cls/super/name/call/subscript).
    - callee_attr is the *final attribute* being called (foo in self.x.foo()).
    """
    if node is None:
        return "unknown", None, None, "name_call"

    if isinstance(node, ast.Attribute):
        # root metadata comes from the value, NOT from the attribute itself
        root_kind, root_name, _, _ = _call_metadata(node.value)
        return root_kind, root_name, node.attr, "attr_call"

    if isinstance(node, ast.Name):
        if node.id == "self":
            return "self", "self", None, "name_call"
        if node.id == "cls":
            return "cls", "cls", None, "name_call"
        return "name", node.id, None, "name_call"

    if isinstance(node, ast.Call):
        # super() special-case
        if isinstance(node.func, ast.Name) and node.func.id == "super":
            return "super", "super", None, "name_call"
        # otherwise it is a call-root like factory() / builder()
        fn = _call_name(node.func)
        return "call", fn, None, "name_call"

    if isinstance(node, ast.Subscript):
        root_kind, root_name, _, _ = _call_metadata(node.value)
        return "subscript", root_name, None, "name_call"

    return "unknown", None, None, "name_call"


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _base_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr
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


def _format_signature(node: ast.FunctionDef) -> str | None:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = ""
    return f"{node.name}({args})" if args else f"{node.name}()"


def parse_module(path: Path) -> ModuleInfo:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _SymbolVisitor()
    visitor.visit(tree)
    return ModuleInfo(path=path, imports=visitor.imports, classes=visitor.classes, functions=visitor.functions)


def parse_python_files(paths: Iterable[Path]) -> List[ModuleInfo]:
    return [parse_module(path) for path in paths]
