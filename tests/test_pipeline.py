"""Pipeline smoke test.

Instructions:
- Run via `pytest tests/test_pipeline.py` to ensure the pipeline completes and writes a report.

Explanation:
- Constructs a temporary estimator class, runs the pipeline over it, and asserts entrypoints and report existence.
"""

from pathlib import Path

from algotracer.pipeline import AlgoTracerPipeline, PipelineConfig


def test_pipeline_runs(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sample = src_dir / "model.py"
    sample.write_text(
        """
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X
""".strip(),
        encoding="utf-8",
    )

    cfg = PipelineConfig(sources=[src_dir], report_dir=tmp_path / "reports")
    pipeline = AlgoTracerPipeline(cfg)
    artifacts = pipeline.run()

    assert artifacts.entrypoints
    assert artifacts.report is not None
    assert artifacts.report.path.exists()
