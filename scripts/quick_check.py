"""Run the most useful local checks for evaluators in one command."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"[CHECK] {name}")
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def check_case_catalog() -> None:
    cases = json.loads((ROOT / "server" / "data" / "cases.json").read_text(encoding="utf-8"))
    difficulties = {case["difficulty"] for case in cases}
    required = {"easy", "medium", "hard"}
    if len(cases) < 3:
        raise SystemExit("Expected at least 3 cases in server/data/cases.json")
    if not required.issubset(difficulties):
        raise SystemExit(
            f"Expected difficulties {sorted(required)}, found {sorted(difficulties)}"
        )
    print(f"[CHECK] case catalog ok: {len(cases)} cases, difficulties={sorted(difficulties)}")


def main() -> None:
    check_case_catalog()
    run_step("openenv validate", ["uv", "run", "openenv", "validate", "."])
    run_step("pytest", ["uv", "run", "pytest"])
    run_step("ruff", ["uv", "run", "ruff", "check", "."])
    run_step("baseline", ["uv", "run", "python", "baseline.py", "--json"])
    print("[CHECK] all local evaluator checks passed")


if __name__ == "__main__":
    main()
