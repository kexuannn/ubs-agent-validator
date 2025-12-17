from __future__ import annotations

"""Entrypoint detection with configurable heuristics + optional auto-discovery.

Two modes:

1) Rules-based (controlled via EntryPointRules):
   - find_entrypoints(modules, names=None, rules=None, add_sklearn_defaults=False)

2) Auto mode (zero-config-ish):
   - find_entrypoints_auto(modules, graph, top_k=10)

Auto mode ranks likely entrypoints using cheap static signals:
- public-looking names (no leading underscore)
- graph out-degree (calls a lot of things)
- number of internal symbol calls (sym -> sym)
- mild bonuses for common entry-ish names (main/run/fit/handle/etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from algotracer.ingest.ast_parser import ClassInfo, ModuleInfo
from algotracer.analysis.deps import DependencyGraph


# ----------------------------
# Data models
# ----------------------------

@dataclass
class EntryPoint:
    module: str          # raw path string
    qualname: str        # e.g. "LogisticRegression.fit"
    lineno: int
    kind: str            # "function" | "method"
    score: int = 0       # ranking score

    @property
    def sym_id(self) -> str:
        return f"sym:{self.module}:{self.qualname}"


@dataclass
class EntryPointRules:
    target_names: Set[str] = field(default_factory=set)
    base_class_hints: Set[str] = field(default_factory=set)
    decorator_hints: Set[str] = field(default_factory=set)
    name_priority: Dict[str, int] = field(default_factory=dict)
    base_bonus: int = 40
    decorator_bonus: int = 25
    default_score: int = 0
    base_requires_name_match: bool = True  # if True, base hints only boost matching names

    # When base_requires_name_match=False, methods in hinted classes can be included even if not target names.
    # This bonus applies to those "base-only" methods (keep smaller than base_bonus to avoid flooding).
    base_only_bonus: int = 10


# ----------------------------
# Rules-based mode
# ----------------------------

def _default_rules(names: Set[str], add_sklearn_defaults: bool = False) -> EntryPointRules:
    # Domain-agnostic defaults unless explicitly asked to add sklearn-like hints.
    target_names = {n.lower() for n in names}
    base_hints: Set[str] = set()
    decorator_hints: Set[str] = set()
    name_priority = {n.lower(): 20 for n in names}

    if add_sklearn_defaults:
        base_hints |= {"baseestimator", "regressormixin", "classifiermixin"}
        target_names |= {"fit", "predict", "transform", "predict_proba"}
        name_priority.update({"fit": 50, "predict": 40, "transform": 35, "predict_proba": 25})

    return EntryPointRules(
        target_names=target_names,
        base_class_hints=base_hints,
        decorator_hints=decorator_hints,
        name_priority=name_priority,
    )


def _match_names(name: str, candidates: Set[str]) -> bool:
    return name.lower() in candidates


def _class_matches(cls: ClassInfo, base_hints: Set[str]) -> bool:
    bases = {b.lower() for b in cls.bases}
    return bool(base_hints & bases)


def _decorator_matches(decorators: Iterable[str], decorator_hints: Set[str]) -> bool:
    lowered = {d.lower() for d in decorators}
    return bool(lowered & decorator_hints)


def _score_name(name: str, rules: EntryPointRules) -> int:
    return rules.name_priority.get(name.lower(), 10)


def find_entrypoints(
    modules: Iterable[ModuleInfo],
    names: Set[str] | None = None,
    rules: EntryPointRules | None = None,
    add_sklearn_defaults: bool = False,
) -> List[EntryPoint]:
    """Rules-based entrypoint detection (good when you know the domain)."""
    if rules is None:
        rules = _default_rules(names or set(), add_sklearn_defaults=add_sklearn_defaults)
    if names:
        rules.target_names |= {n.lower() for n in names}

    matches: List[EntryPoint] = []
    for module in modules:
        module_key = str(module.path)

        # Top-level functions
        for func in module.functions:
            if _match_names(func.name, rules.target_names):
                matches.append(
                    EntryPoint(
                        module=module_key,
                        qualname=func.name,
                        lineno=func.lineno,
                        kind="function",
                        score=rules.default_score + _score_name(func.name, rules),
                    )
                )

        # Class methods
        for cls in module.classes:
            class_hint = _class_matches(cls, rules.base_class_hints)

            for method in cls.methods:
                is_target = _match_names(method.name, rules.target_names)
                has_decorator_hint = _decorator_matches(method.decorators, rules.decorator_hints)

                allow_base_only = class_hint and not rules.base_requires_name_match
                should_include = is_target or has_decorator_hint or allow_base_only
                if not should_include:
                    continue

                qual = f"{cls.name}.{method.name}"

                score = rules.default_score + _score_name(method.name, rules)

                # Full base bonus only when (a) class matches base hints AND (b) method name matches target.
                if class_hint and is_target:
                    score += rules.base_bonus

                # If we're in base-only mode, include base-hinted methods but with a smaller bonus
                if allow_base_only and not is_target:
                    score += rules.base_only_bonus

                if has_decorator_hint:
                    score += rules.decorator_bonus

                matches.append(
                    EntryPoint(
                        module=module_key,
                        qualname=qual,
                        lineno=method.lineno,
                        kind="method",
                        score=score,
                    )
                )

    matches = sorted(matches, key=lambda e: e.score, reverse=True)
    print(f"AlgoTracer: rules-based entrypoints found={len(matches)}")
    return matches


# ----------------------------
# Auto mode (zero-config)
# ----------------------------

_COMMON_ENTRY_NAMES: Dict[str, int] = {
    # general
    "main": 50,
    "run": 35,
    "start": 30,
    "execute": 30,
    "handle": 25,
    "dispatch": 25,
    "process": 20,
    "serve": 25,
    # sklearn-ish (still helpful as weak priors)
    "fit": 35,
    "predict": 25,
    "transform": 20,
    "predict_proba": 15,
    # vn.py-ish/common event handlers
    "on_tick": 25,
    "on_bar": 25,
    "on_trade": 20,
    "on_order": 15,
}


def _is_public_name(qualname: str) -> bool:
    parts = qualname.split(".")
    return all(not p.startswith("_") for p in parts if p)


def _sym_id(module_path: str, qualname: str) -> str:
    return f"sym:{module_path}:{qualname}"


def _graph_features(graph: DependencyGraph, sym_id: str) -> Tuple[int, int, int]:
    """Return (out_degree, internal_sym_neighbors, external_neighbors)."""
    nbrs = graph.neighbors(sym_id)
    out_deg = len(nbrs)
    internal = sum(1 for n in nbrs if n.startswith("sym:"))
    external = sum(1 for n in nbrs if n.startswith("ext:") or n.startswith("mod:"))
    return out_deg, internal, external


def _auto_score(qualname: str, out_deg: int, internal: int, external: int) -> int:
    score = 0

    # public API bias
    score += 15 if _is_public_name(qualname) else -10

    # weak name prior
    leaf = qualname.split(".")[-1].lower()
    score += _COMMON_ENTRY_NAMES.get(leaf, 0)

    # graph influence
    score += 2 * internal
    score += 1 * external
    score += min(out_deg, 50)

    return score


def find_entrypoints_auto(
    modules: Iterable[ModuleInfo],
    graph: DependencyGraph,
    top_k: int = 10,
) -> List[EntryPoint]:
    """Auto-discover likely entrypoints by ranking ALL symbols with generic signals."""
    candidates: List[EntryPoint] = []

    for module in modules:
        module_key = str(module.path)

        # top-level functions
        for func in module.functions:
            qual = func.name
            sid = _sym_id(module_key, qual)
            out_deg, internal, external = _graph_features(graph, sid)
            score = _auto_score(qual, out_deg, internal, external)
            candidates.append(
                EntryPoint(
                    module=module_key,
                    qualname=qual,
                    lineno=func.lineno,
                    kind="function",
                    score=score,
                )
            )

        # class methods
        for cls in module.classes:
            for method in cls.methods:
                qual = f"{cls.name}.{method.name}"
                sid = _sym_id(module_key, qual)
                out_deg, internal, external = _graph_features(graph, sid)
                score = _auto_score(qual, out_deg, internal, external)
                candidates.append(
                    EntryPoint(
                        module=module_key,
                        qualname=qual,
                        lineno=method.lineno,
                        kind="method",
                        score=score,
                    )
                )

    candidates.sort(key=lambda e: e.score, reverse=True)
    result = candidates[:top_k]
    print(f"AlgoTracer: auto entrypoints scored={len(candidates)}, returning top={len(result)}")
    return result
