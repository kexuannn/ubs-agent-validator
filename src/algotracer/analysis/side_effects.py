from __future__ import annotations

"""Heuristic side-effect detection for function/method calls."""

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
    _SideEffectRule(
        category="io.file",
        patterns=("open", "path.open", "io.open"),
        confidence=0.35,
        evidence="heuristic: file open call",
    ),
    _SideEffectRule(
        category="io.file_read",
        patterns=("read", "read_text", "read_bytes", "load", "loadtxt"),
        confidence=0.3,
        evidence="heuristic: file read-ish call",
    ),
    _SideEffectRule(
        category="io.file_write",
        patterns=("write", "write_text", "write_bytes", "save", "dump"),
        confidence=0.35,
        evidence="heuristic: file write-ish call",
    ),
    _SideEffectRule(
        category="io.console",
        patterns=("print", "logging.", "logger."),
        confidence=0.45,
        evidence="heuristic: console/logging call",
    ),
    _SideEffectRule(
        category="io.network",
        patterns=("requests.", "httpx.", "urllib.request.", "urllib3.", "aiohttp.", "socket.", "websocket."),
        confidence=0.65,
        evidence="heuristic: network/HTTP call",
    ),
    _SideEffectRule(
        category="system.process",
        patterns=("subprocess.", "os.system", "os.popen", "os.spawn"),
        confidence=0.7,
        evidence="heuristic: subprocess/system call",
    ),
    _SideEffectRule(
        category="external.service",
        patterns=("boto3.", "google.cloud.", "azure.", "kafka.", "redis."),
        confidence=0.6,
        evidence="heuristic: external service SDK call",
    ),
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
]


def _matches_pattern(call: str, pattern: str) -> bool:
    call_l = call.lower()
    pattern_l = pattern.lower()
    if pattern_l.endswith("."):
        return call_l.startswith(pattern_l)
    if "." in pattern_l:
        return call_l == pattern_l or call_l.startswith(pattern_l + ".")
    return call_l == pattern_l or call_l.endswith("." + pattern_l)


def infer_side_effects(calls: Iterable[str]) -> List[SideEffect]:
    """Return side-effect annotations inferred from call names."""
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
