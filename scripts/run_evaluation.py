"""
scripts/run_evaluation.py
──────────────────────────
Run the clinical assistant evaluation framework against benchmark cases.

Usage:
    python scripts/run_evaluation.py                          # all cases, all metrics
    python scripts/run_evaluation.py --case depression_001    # single case
    python scripts/run_evaluation.py --json                   # JSON output
    python scripts/run_evaluation.py --output results.json    # save to file

The script runs the full LangGraph pipeline for each case, then computes
all 7 evaluation metrics. Metrics are defined in clinical/evaluation/metrics/.

Requires: all pipeline dependencies (Ollama, ChromaDB, embedding model).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os; os.environ["LOG_FORMAT"] = "console"

from app_logging.logger import setup_logging, get_logger
from clinical.evaluation import ALL_CASES, evaluate_all, format_report, format_json_report
from clinical.evaluation.benchmarks.cases import BenchmarkCase

setup_logging(level="INFO", log_format="console")
_log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clinical Psychology Assistant — Evaluation Framework",
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Run a single case by ID (e.g. depression_001)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to a JSON file",
    )
    args = parser.parse_args()

    cases: list[BenchmarkCase] = ALL_CASES
    if args.case:
        filtered = [c for c in cases if c.case_id == args.case]
        if not filtered:
            _log.error("eval.case_not_found", case_id=args.case)
            sys.exit(1)
        cases = filtered

    _log.info(
        "eval.start",
        case_count=len(cases),
        cases=[c.case_id for c in cases],
    )

    t_start = time.perf_counter()

    import asyncio
    report = asyncio.run(evaluate_all(cases=cases))

    elapsed_s = time.perf_counter() - t_start

    if args.json or args.output:
        data = format_json_report(report)
        data["elapsed_s"] = round(elapsed_s, 2)
        output = json.dumps(data, indent=2, default=str)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            _log.info("eval.saved", path=args.output)
        else:
            print(output)
    else:
        print(format_report(report))
        print(f"\n  Total evaluation time: {elapsed_s:.1f}s")

    _log.info(
        "eval.complete",
        case_count=len(cases),
        elapsed_s=round(elapsed_s, 2),
    )


if __name__ == "__main__":
    main()
