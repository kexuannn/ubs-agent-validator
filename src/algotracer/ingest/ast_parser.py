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

NEW:
- Lightweight type metadata:
  - FunctionInfo.param_type_hints: raw annotation strings for params
  - FunctionInfo.local_type_hints: raw annotation strings for locals
  - FunctionInfo.local_type_inferred: coarse type strings from value patterns
  - ClassInfo.field_types: type strings for self/cls attributes (e.g. self.pos_data)
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set


# ============================================================================
# Data models for imports, calls, functions, classes, modules
# ============================================================================


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
    end_lineno: int | None = None
    signature: str | None = None
    docstring: str | None = None
    params: Set[str] = field(default_factory=set)
    assigned_locals: Set[str] = field(default_factory=set)  # local variable names
    calls: Set[str] = field(default_factory=set)
    call_details: List[CallInfo] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    # lightweight type metadata for params & locals
    param_type_hints: dict[str, str] = field(default_factory=dict)
    local_type_hints: dict[str, str] = field(default_factory=dict)
    local_type_inferred: dict[str, str] = field(default_factory=dict)
    # aliases where a local is assigned from a self/cls field:
    #   target = self.chart_item  ->  self_aliases["target"] = "chart_item"
    self_aliases: dict[str, str] = field(default_factory=dict)



@dataclass
class ClassInfo:
    name: str
    lineno: int
    end_lineno: int | None = None
    bases: List[str] = field(default_factory=list)  # list of base class names (for inheritance tracing)
    methods: List[FunctionInfo] = field(default_factory=list)
    # NEW: inferred/annotated types for self/cls attributes on this class
    field_types: dict[str, str] = field(default_factory=dict)
    docstring: str | None = None


@dataclass
class ModuleInfo:  # a single .py file
    path: Path
    imports: List[ImportInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)  # top-level function info (not inside classes)


# ============================================================================
# Symbol visitor: collects imports/classes/functions and call metadata
# ============================================================================


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: List[ImportInfo] = []  # imports
        self.classes: List[ClassInfo] = []  # top-level classes
        self.functions: List[FunctionInfo] = []  # top-level functions
        self._class_stack: List[ClassInfo] = []  # track current nested class context

    def _current_class_name(self) -> str | None:
        return self._class_stack[-1].name if self._class_stack else None

    # -----------------------------------------------------------------------
    # Imports
    # -----------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:  # 'import x' / 'import x as y'
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

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # 'from ... import ...'
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

    # -----------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        decos: List[str] = []
        for d in node.decorator_list:
            name = _decorator_name(d)
            if name:
                decos.append(name)

        params, param_type_hints = _collect_params(node.args)

        func = FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            signature=_format_signature(node),
            decorators=decos,
            params=params,
        )
        func.docstring = ast.get_docstring(node)
        func.param_type_hints = param_type_hints

        calls, details = _collect_calls(node, current_class=self._current_class_name())
        func.calls = calls
        func.call_details = details

        locals_names, local_type_hints, local_type_inferred, self_aliases = _collect_locals(node)
        func.assigned_locals = locals_names
        func.local_type_hints = local_type_hints
        func.local_type_inferred = local_type_inferred
        func.self_aliases = self_aliases


        if self._class_stack:
            self._class_stack[-1].methods.append(func)
        else:
            self.functions.append(func)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    # -----------------------------------------------------------------------
    # Classes
    # -----------------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Collect self/cls attribute type info for this class
        field_collector = _SelfFieldCollector()
        field_collector.visit(node)

        cls = ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            bases=[_base_name(base) for base in node.bases],
        )
        cls.field_types = field_collector.field_types
        cls.docstring = ast.get_docstring(node)

        self._class_stack.append(cls)
        self.generic_visit(node)
        self._class_stack.pop()
        self.classes.append(cls)


# ============================================================================
# Call collector: captures normalized calls and detailed metadata
# ============================================================================


class _CallCollector(ast.NodeVisitor):
    def __init__(self, current_class: str | None) -> None:
        self.calls: Set[str] = set()
        self.call_details: List[CallInfo] = []
        self._current_class = current_class

    def visit_Call(self, node: ast.Call) -> None:
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


# ============================================================================
# Collection helpers: calls, params, locals & lightweight type info
# ============================================================================


def _collect_calls(node: ast.AST, current_class: str | None) -> tuple[Set[str], List[CallInfo]]:
    collector = _CallCollector(current_class)
    collector.visit(node)
    return collector.calls, collector.call_details


def _annotation_to_str(node: ast.AST | None) -> str | None:
    """Best-effort string form of a type annotation expression."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _value_type_expr(node: ast.AST | None) -> str | None:
    """
    Very coarse type inference from a value expression.

    Returns a simple type expression string like:
      - "dict", "list", "set", "tuple"
      - "SomeClass" or "pkg.SomeClass" (for constructor calls)
    """
    if node is None:
        return None

    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"

    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name):
            return fn.id
        if isinstance(fn, ast.Attribute):
            return _base_name(fn)

    return None


def _collect_params(args: ast.arguments) -> tuple[Set[str], dict[str, str]]:
    """
    Collect parameter names AND any type annotations as raw strings.
    """
    params: Set[str] = set()
    type_hints: dict[str, str] = {}

    def _handle_arg(arg: ast.arg | None) -> None:
        if arg is None:
            return
        params.add(arg.arg)
        if arg.annotation is not None:
            t = _annotation_to_str(arg.annotation)
            if t:
                type_hints[arg.arg] = t

    for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
        _handle_arg(a)

    _handle_arg(args.vararg)
    _handle_arg(args.kwarg)

    return params, type_hints


class _LocalCollector(ast.NodeVisitor):
    """
    Collect names assigned within a function body (excluding nested defs),
    plus lightweight type hints / inference for locals.

    NEW:
    - self_aliases: map local -> self/cls field name when we see:
          target = self.field
          alias  = cls.registry
      This lets the builder treat `target.update(...)` like `self.field.update(...)`.
    """

    def __init__(self) -> None:
        self.locals: Set[str] = set()
        self.type_hints: dict[str, str] = {}
        self.inferred_types: dict[str, str] = {}
        self.self_aliases: dict[str, str] = {}  # local_name -> field_name

    def _record_targets(self, targets: Iterable[ast.AST]) -> None:
        for tgt in targets:
            self._record_target(tgt)

    def _record_target(self, tgt: ast.AST) -> None:
        if isinstance(tgt, ast.Name):
            self.locals.add(tgt.id)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            self._record_targets(tgt.elts)

    def _maybe_record_self_alias(self, node: ast.Assign) -> None:
        """
        Detect simple aliasing patterns like:
            target = self.field
            alias  = cls.registry
        and record: self_aliases["target"] = "field".
        """
        if len(node.targets) != 1:
            return
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name):
            return
        if not isinstance(value, ast.Attribute):
            return
        if not isinstance(value.value, ast.Name):
            return
        if value.value.id not in {"self", "cls"}:
            return

        local_name = target.id
        field_name = value.attr
        # Don't overwrite if we already have a mapping
        self.self_aliases.setdefault(local_name, field_name)

    def visit_Assign(self, node: ast.Assign) -> None:  # normal assignments; 'x = expr'
        self._record_targets(node.targets)

        # Track aliases like: target = self.field / cls.field
        self._maybe_record_self_alias(node)

        # Simple inference for single-target assignments
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            t = _value_type_expr(node.value)
            if t:
                # don't overwrite an existing explicit hint
                self.inferred_types.setdefault(name, t)

        self.generic_visit(node.value)  # traverse value to catch further things inside

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # 'x: int = 1'
        self._record_target(node.target)

        if isinstance(node.target, ast.Name):
            name = node.target.id
            t = _annotation_to_str(node.annotation)
            if t:
                self.type_hints[name] = t

        if node.value:
            self.generic_visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # 'x += 1'
        self._record_target(node.target)
        self.generic_visit(node.value)

    def visit_For(self, node: ast.For) -> None:  # 'for' loops
        self._record_target(node.target)
        self.generic_visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:  # 'with' statements
        for item in node.items:
            if item.optional_vars:
                self._record_target(item.optional_vars)
            self.visit(item.context_expr)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # 'x := expr'
        self._record_target(node.target)
        self.generic_visit(node.value)

    # Do not descend into nested defs; keep locals scoped to the current function body.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _collect_locals(func_node: ast.AST) -> tuple[Set[str], dict[str, str], dict[str, str], dict[str, str]]:
    collector = _LocalCollector()
    for stmt in getattr(func_node, "body", []):
        collector.visit(stmt)
    return collector.locals, collector.type_hints, collector.inferred_types, collector.self_aliases



class _SelfFieldCollector(ast.NodeVisitor):
    """
    Collects self/cls attribute type information for a class.

    We look for patterns like:
        self.pos_data: dict[str, int] = {}
        self.client = httpx.Client(...)
    and record a simple type expression string per field name.
    """

    def __init__(self) -> None:
        self.field_types: dict[str, str] = {}

    def _record_self_attr(
        self,
        target: ast.AST,
        value: ast.AST | None,
        annotation: ast.AST | None = None,
    ) -> None:
        if not isinstance(target, ast.Attribute):
            return
        if not isinstance(target.value, ast.Name):
            return
        if target.value.id not in {"self", "cls"}:
            return

        name = target.attr
        t = None
        if annotation is not None:
            t = _annotation_to_str(annotation)
        if not t and value is not None:
            t = _value_type_expr(value)

        if t:
            # Don't overwrite an existing explicit annotation
            self.field_types.setdefault(name, t)

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            self._record_self_attr(tgt, node.value, None)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_self_attr(node.target, node.value, node.annotation)
        if node.value:
            self.generic_visit(node.value)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        When called on the *current* class, manually walk its body but do not recurse
        into nested classes (those will be handled by their own ClassInfo).
        """
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                # nested class – skip here, will be handled by its own ClassInfo
                continue
            self.visit(stmt)


# ============================================================================
# Call normalization and metadata helpers
# ============================================================================


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


def _base_name(node: ast.expr) -> str:  # normalises base class expressions in 'class X(Base)'
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _base_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    return ast.unparse(node)


# ============================================================================
# Decorator/signature helpers
# ============================================================================


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


# ============================================================================
# Public API
# ============================================================================


def parse_module(path: Path) -> ModuleInfo:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _SymbolVisitor()
    visitor.visit(tree)
    return ModuleInfo(path=path, imports=visitor.imports, classes=visitor.classes, functions=visitor.functions)


def parse_python_files(paths: Iterable[Path]) -> List[ModuleInfo]:
    return [parse_module(path) for path in paths]
