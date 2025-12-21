from __future__ import annotations

"""Heuristic side-effect detection for function/method calls.

This module focuses on *externally visible* side effects and cross-cutting runtime
behaviour, not generic in-memory mutation.

We tag calls in categories such as:
- io.file / io.file_read / io.file_write
- io.console / io.network
- system.process
- external.service
- db.query / db.transaction
- runtime.time / runtime.random
- runtime.global_state

Common list/dict/DataFrame mutators (append, pop, drop, etc.) are *not* treated as
strong side effects here; they are handled separately by the explainer as a softer
“may imply state changes” hint.
"""

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class SideEffect:
    category: str
    call: str
    confidence: float
    evidence: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "call": self.call,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class _SideEffectRule:
    category: str
    patterns: tuple[str, ...]
    confidence: float
    evidence: str


_RULES: List[_SideEffectRule] = [
    # -------- Filesystem I/O --------
    _SideEffectRule(
        category="io.file",
        patterns=(
            "open",              # builtins.open
            "path.open",         # Path.open (generic)
            "pathlib.Path.open", # explicit pathlib
            "io.open",
        ),
        confidence=0.35,
        evidence="heuristic: file open call",
    ),
    _SideEffectRule(
        category="io.file_read",
        patterns=(
            "read",
            "read_text",
            "read_bytes",
            "load",
            "loadtxt",
        ),
        confidence=0.3,
        evidence="heuristic: file read-ish call",
    ),
    _SideEffectRule(
        category="io.file_write",
        patterns=(
            "write",
            "write_text",
            "write_bytes",
            "save",
            "dump",
        ),
        confidence=0.35,
        evidence="heuristic: file write-ish call",
    ),

    # -------- Console / logging --------
    _SideEffectRule(
        category="io.console",
        patterns=(
            "print",     # built-in print
            "logging.",  # logging.info/debug/error/...
            "logger.",   # custom logger.* calls
        ),
        confidence=0.45,
        evidence="heuristic: console/logging call",
    ),

    # -------- Network / HTTP --------
    _SideEffectRule(
        category="io.network",
        patterns=(
            "requests.",
            "httpx.",
            "urllib.request.",
            "urllib3.",
            "aiohttp.",
            "socket.",
            "websocket.",
        ),
        confidence=0.65,
        evidence="heuristic: network/HTTP call",
    ),

    # -------- Subprocess / system process --------
    _SideEffectRule(
        category="system.process",
        patterns=(
            "subprocess.",
            "os.system",
            "os.popen",
            "os.spawn",
        ),
        confidence=0.7,
        evidence="heuristic: subprocess/system call",
    ),

    # -------- External services / infra SDKs --------
    _SideEffectRule(
        category="external.service",
        patterns=(
            "boto3.",
            "google.cloud.",
            "azure.",
            "kafka.",
            "confluent_kafka.",
            "redis.",
        ),
        confidence=0.6,
        evidence="heuristic: external service SDK call",
    ),

    # -------- Database / queries / ORM --------
    _SideEffectRule(
        category="db.query",
        patterns=(
            "cursor.execute",
            "cursor.executemany",
            "cursor.fetchone",
            "cursor.fetchall",
            "session.query",
            "sqlalchemy.",
        ),
        confidence=0.55,
        evidence="heuristic: database/query call",
    ),
    _SideEffectRule(
        category="db.transaction",
        patterns=(
            "session.commit",
            "session.rollback",
            "transaction.commit",
            "transaction.rollback",
        ),
        confidence=0.6,
        evidence="heuristic: database transaction control",
    ),

    # -------- Time / non-determinism --------
    _SideEffectRule(
        category="runtime.time",
        patterns=(
            "time.time",
            "time.sleep",
            "datetime.now",
            "datetime.utcnow",
            "datetime.datetime.now",
            "datetime.datetime.utcnow",
        ),
        confidence=0.4,
        evidence="heuristic: time-based non-determinism",
    ),

    # -------- Randomness / non-determinism --------
    _SideEffectRule(
        category="runtime.random",
        patterns=(
            "random.",
            "secrets.",
            "numpy.random.",
            "np.random.",
        ),
        confidence=0.45,
        evidence="heuristic: random-based non-determinism",
    ),

    # -------- Global / configuration / environment --------
    _SideEffectRule(
        category="runtime.global_state",
        patterns=(
            "os.environ",
            "settings.",
            "config.",
            "django.conf.settings",
        ),
        confidence=0.5,
        evidence="heuristic: global/config/env mutation or access",
    ),
]


def _matches_pattern(call: str, pattern: str) -> bool:
    """Return True if `call` matches the heuristic `pattern`.

    Rules (case-insensitive):
    - If pattern ends with '.', treat it as a prefix: e.g. 'logging.' matches 'logging.info'.
    - If pattern contains '.', match the exact name or a dotted prefix:
        'requests.get' matches 'requests.get' and 'requests.get.json'.
    - Otherwise, match the exact name or a common suffix:
        'open' matches 'open' and 'io.open'.
    """
    call_l = call.lower()
    pattern_l = pattern.lower()
    if pattern_l.endswith("."):
        return call_l.startswith(pattern_l)
    if "." in pattern_l:
        return call_l == pattern_l or call_l.startswith(pattern_l + ".")
    return call_l == pattern_l or call_l.endswith("." + pattern_l)


def infer_side_effects(calls: Iterable[str]) -> List[SideEffect]:
    """Return side-effect annotations inferred from call names.

    `calls` are fully-qualified or relative call names as produced by the analyzer
    (e.g. 'logging.info', 'requests.get', 'cursor.execute').

    Each returned SideEffect is de-duplicated by (category, call).
    """
    seen: set[tuple[str, str]] = set()
    effects: List[SideEffect] = []
    for call in calls:
        for rule in _RULES:
            if any(_matches_pattern(call, pattern) for pattern in rule.patterns):
                key = (rule.category, call)
                if key in seen:
                    continue
                seen.add(key)
                effects.append(
                    SideEffect(
                        category=rule.category,
                        call=call,
                        confidence=rule.confidence,
                        evidence=rule.evidence,
                    )
                )
    return effects
