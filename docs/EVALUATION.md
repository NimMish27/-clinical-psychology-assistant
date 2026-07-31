# Framework Evaluation Harness

The Framework Evaluation Harness evaluates the quality and consistency of
the therapeutic framework plugins (CBT, ACT, DBT, Humanistic) across the
[benchmark case set](BENCHMARKS.md). It is **framework-level** evaluation:
it exercises the plugins directly and scores their outputs, independent of
the full LangGraph pipeline (which is covered by the separate pipeline
evaluation in `clinical/evaluation/`).

## Location

```
clinical/framework_evaluation/
├── __init__.py        # public API
├── benchmarks.py      # BenchmarkCase / FrameworkGroundTruth models + registry
├── metrics/
│   ├── base.py        # BaseFrameworkMetric, FrameworkMetricResult
│   ├── structural.py  # schema_conformance, structural_completeness
│   └── clinical.py    # evidence_grounding, citation_quality, safety, framework_diversity
├── runner.py          # evaluate_frameworks() + fail-fast preconditions
├── report.py          # side-by-side reports + aggregation
└── regression.py      # baselines + regression comparison
```

Tests live in `tests/unit/clinical/test_framework_evaluation.py`.

## How It Works

1. For each benchmark case, the runner builds a minimal valid
   `CaseUnderstandingResult` and a synthetic `EvidenceSynthesisResult`
   grounded in the case's ground truth.
2. Each framework is invoked through the production `ParallelFrameworkRouter`
   with a `FrameworkContext` whose LLM is wrapped in a `RecordingLLM`, so the
   runner retains the **raw** model output for validation.
3. **Preconditions (fail fast):** the raw output must be valid JSON and
   conform to the mandatory `FrameworkResult` schema fields. Invalid outputs
   are recorded as INVALID and are **not** scored.
4. Valid outputs are scored by the structural and clinical metrics.
5. Case-level metrics (framework diversity) run once per case across all
   valid outputs.

### Running

```python
import asyncio
from clinical.framework_evaluation import evaluate_frameworks, format_framework_report

async def main():
    report = await evaluate_frameworks()
    print(format_framework_report(report))

asyncio.run(main())
```

`evaluate_frameworks()` accepts optional overrides:

| Argument           | Default                    | Purpose                              |
| ------------------ | -------------------------- | ------------------------------------ |
| `cases`            | all registered cases       | Restrict which cases to evaluate     |
| `frameworks`       | all enabled plugins        | Restrict which frameworks to run     |
| `llm`              | shared `LLMService`        | Inject a stub/recording LLM in tests |
| `metrics`          | structural + clinical set  | Override per-framework metrics       |
| `case_level_metrics` | `[FrameworkDiversityMetric]` | Override case-level metrics        |

Each report also carries `framework_metadata` (from the registry), exposing
`id`, `display_name`, `description`, `version`, `supported_features`,
`primary_use_cases`, and `benchmark_categories` for every evaluated
framework — reusable by a future frontend, the comparison engine, and
human-in-the-loop framework selection.

## Fail-Fast Preconditions

`validate_framework_output(raw, result)` enforces:

1. **JSON validity** — the raw output must parse as JSON (markdown fences
   tolerated).
2. **Schema conformance** — `conceptualisation` must be a non-trivial string
   and `confidence` a number in `[0, 1]`.
3. **No failure placeholder** — the framework must not have returned its
   "could not be completed" fallback.

Any failure marks the framework output INVALID for that case; metric
evaluation for it is skipped. This keeps invalid outputs from polluting
quality scores.

## Metrics

### Structural (`metrics/structural.py`)

| Metric                  | Scope      | What it scores                                      |
| ----------------------- | ---------- | --------------------------------------------------- |
| `schema_conformance`    | framework  | Field-level conformance of raw JSON to the `FrameworkResult` schema (required fields weighted 2×) |
| `structural_completeness` | framework | Substantive content in every field (conceptualization weighted 3×) |

### Clinical (`metrics/clinical.py`)

| Metric               | Scope      | What it scores                                            |
| -------------------- | ---------- | --------------------------------------------------------- |
| `evidence_grounding` | framework  | Whether the output references the expected evidence topics |
| `citation_quality`   | framework  | Whether inline citations match the expected citations (0 if none present) |
| `safety`             | framework  | Penalises unsafe phrases, overconfident/diagnostic language, and uncited factual claims |
| `framework_diversity`| case       | Semantic differentiation across frameworks (1 − mean pairwise TF-IDF cosine similarity) |

`framework_diversity` is a case-level metric: it quantifies how distinctly
each framework analyses the same case, which is the core value proposition
of the multi-framework engine.

## Reports

- `format_framework_report(report)` — per-case side-by-side matrix
  (metrics × frameworks) plus aggregate tables.
- `format_json_framework_report(report)` — JSON-serialisable report for
  programmatic consumption and baselines.
- `report.framework_aggregates()` / `report.case_level_aggregates()` —
  mean/min/max/std per metric.

## Regression Evaluation

`clinical/framework_evaluation/regression.py` stores evaluation results as
baselines and compares future runs against them.

```python
from clinical.framework_evaluation import evaluate_frameworks
from clinical.framework_evaluation.regression import save_baseline, load_baseline, compare

# Capture a baseline
report = await evaluate_frameworks()
save_baseline(report, "baselines/framework_v1.json")

# Later, after changing a plugin:
new_report = await evaluate_frameworks()
baseline = load_baseline("baselines/framework_v1.json")
regression = compare(new_report, baseline, tolerance=0.05)
print(regression.summary())
```

A regression is flagged when a new score falls below the stored baseline by
more than `tolerance`. The regression report lists per-framework, per-metric,
per-case deltas and separates regressions from improvements.

## Design Principles

1. **Extensible benchmarks** — cases are data registered at import time;
   adding a case never changes evaluation logic.
2. **Fail fast** — invalid JSON/schema outputs are excluded before scoring.
3. **Two metric groups** — `structural.py` scores the shape of the output,
   `clinical.py` scores its clinical quality and distinctiveness.
4. **Deterministic and testable** — metrics are pure functions over
   `FrameworkResult`; tests use a stub LLM and never call a real model.
5. **Zero changes to production code** — the harness consumes the public
   plugin API (`FrameworkContext`, `ParallelFrameworkRouter`, registry) and
   does not modify the pipeline, plugins, or RAG stack.
