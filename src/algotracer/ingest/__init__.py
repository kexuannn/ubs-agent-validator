"""AST ingestion utilities.

Instructions:
- Import parse_module or parse_python_files to turn Python sources into ModuleInfo structures.
- Re-export ModuleInfo for downstream consumers.

Explanation:
- Provides convenient access to ingestion helpers without deep import paths.
"""

from .ast_parser import parse_python_files, parse_module, ModuleInfo

__all__ = ["parse_python_files", "parse_module", "ModuleInfo"]
