"""Reporting helpers for AlgoTracer.

Instructions:
- Import render_markdown_report to generate Markdown output from analysis artifacts.
- Use MarkdownReport for downstream consumers if needed.

Explanation:
- Groups reporting utilities for easier imports and future expansion (HTML, diagrams).
"""

from .renderer import MarkdownReport, render_markdown_report

__all__ = ["MarkdownReport", "render_markdown_report"]
