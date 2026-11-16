"""Import boundary: agent and tool code may not reach private modules.

Agent and tool code may import `causal_agent.bench.public` and nothing else
from `causal_agent.bench`, and nothing from `causal_agent.eval`. The scan
passes vacuously until those packages exist and becomes binding once they do.
"""

import ast
from pathlib import Path

import causal_agent

PACKAGE_ROOT = Path(causal_agent.__file__).parent
RESTRICTED_PACKAGES = ("agent", "tools")
ALLOWED_BENCH_MODULE = "causal_agent.bench.public"


def forbidden_imports(source: str) -> list[str]:
    """Return the private modules a piece of source code imports."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                modules = [f"<relative import of {node.module or '.'}>"]
            elif node.module == "causal_agent.bench":
                modules = [f"causal_agent.bench.{alias.name}" for alias in node.names]
            elif node.module == "causal_agent":
                modules = [f"causal_agent.{alias.name}" for alias in node.names]
            else:
                modules = [node.module or ""]
        else:
            continue
        for module in modules:
            private_bench = (
                module == "causal_agent.bench"
                or module.startswith("causal_agent.bench.")
            ) and module != ALLOWED_BENCH_MODULE
            private_eval = module == "causal_agent.eval" or module.startswith(
                "causal_agent.eval."
            )
            if private_bench or private_eval or module.startswith("<relative"):
                found.append(module)
    return found


def test_scanner_allows_the_public_module() -> None:
    assert forbidden_imports("from causal_agent.bench.public import load_public") == []
    assert forbidden_imports("import causal_agent.bench.public") == []
    assert forbidden_imports("from causal_agent.bench import public") == []
    assert forbidden_imports("import numpy as np\nfrom pathlib import Path") == []


def test_scanner_catches_private_imports() -> None:
    assert forbidden_imports("from causal_agent.bench.truth import load_truth")
    assert forbidden_imports("import causal_agent.bench.spec")
    assert forbidden_imports("from causal_agent.bench import truth")
    assert forbidden_imports("from causal_agent.bench import public, generate")
    assert forbidden_imports("import causal_agent.bench")
    assert forbidden_imports("from causal_agent import eval")
    assert forbidden_imports("from causal_agent.eval.score import score")
    assert forbidden_imports("from ..bench import truth")


def test_agent_and_tool_code_import_no_private_module() -> None:
    violations: dict[str, list[str]] = {}
    for package in RESTRICTED_PACKAGES:
        for path in sorted((PACKAGE_ROOT / package).rglob("*.py")):
            found = forbidden_imports(path.read_text(encoding="utf-8"))
            if found:
                violations[str(path.relative_to(PACKAGE_ROOT))] = found
    assert violations == {}
