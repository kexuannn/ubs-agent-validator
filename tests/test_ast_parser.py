"""Tests for AST parser.

Instructions:
- Run via `pytest tests/test_ast_parser.py` to validate parsing of imports, classes, and functions.

Explanation:
- Builds a temporary module with a sklearn-style class and ensures parsed symbols are present.
"""

from pathlib import Path

from algotracer.ingest.ast_parser import parse_module


def test_parse_module(tmp_path: Path) -> None:
    code = """
import numpy as np
from sklearn.base import BaseEstimator

class LinearModel(BaseEstimator):
    def fit(self, X, y):
        return np.linalg.lstsq(X, y)


def helper(x):
    return x.mean()
"""
    target = tmp_path / "sample.py"
    target.write_text(code.strip(), encoding="utf-8")

    module = parse_module(target)
    assert module.imports
    assert any(cls.name == "LinearModel" for cls in module.classes)
    assert any(func.name == "fit" for cls in module.classes for func in cls.methods)
    assert any(func.name == "helper" for func in module.functions)
