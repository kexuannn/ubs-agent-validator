"""Reasoning helpers for structural traces.

Instructions:
- Import explain_entrypoints to get per-entrypoint reasoning strings (LLM-backed if provided).
- Import build_trace_prompt if you want to call an LLM yourself.
- Import build_gemini_llm to use Gemini via google-generativeai.

Explanation:
- Groups reasoning utilities for structural trace summaries.
"""

from .flow_explainer import explain_entrypoints, build_trace_prompt, build_gemini_llm

__all__ = ["explain_entrypoints", "build_trace_prompt", "build_gemini_llm"]
