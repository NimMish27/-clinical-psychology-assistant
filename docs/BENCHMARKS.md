# Framework Evaluation Benchmarks

This document describes the benchmark case set used by the
[Framework Evaluation Harness](EVALUATION.md) to evaluate the quality and
consistency of the therapeutic framework plugins (CBT, ACT, Humanistic).

## Overview

Benchmark cases live in `clinical/framework_evaluation/benchmarks.py`.
They are **data, not code** — the evaluation logic is entirely
category-agnostic and reads the ground truth generically. Adding a new
benchmark case never requires changing the metrics, runner, or report code.

## Data Model

### `FrameworkGroundTruth`

Framework-scoped expectations for a case:

| Field                  | Type            | Purpose                                              |
| ---------------------- | --------------- | ---------------------------------------------------- |
| `expected_topics`      | `list[str]`     | Clinical topics a competent analysis should touch on |
| `expected_evidence_topics` | `list[str]` | Evidence themes the framework outputs should reference |
| `expected_citations`   | `list[str]`     | Citations a well-grounded output should include inline |
| `unsafe_phrases`       | `list[str]`     | Language the output must avoid (diagnosis, prescription, absolute claims) |
| `differentiation_notes`| `dict[str, str]`| Per-framework notes describing the expected analytical lens |
| `metadata`             | `dict[str, Any]`| Reserved for future case-specific ground truth       |

### `BenchmarkCase`

| Field          | Type               | Purpose                                              |
| -------------- | ------------------ | ---------------------------------------------------- |
| `case_id`      | `str`              | Unique identifier (e.g. `burnout_001`)               |
| `title`        | `str`              | Human-readable title                                 |
| `clinical_text`| `str`              | The case study the frameworks analyse                |
| `input_type`   | `str`              | Input kind (defaults to `case_study`)                |
| `categories`   | `list[str]`        | Flexible grouping tags — **never** used for scoring  |
| `ground_truth` | `FrameworkGroundTruth` | Framework-scoped expectations                    |
| `metadata`     | `dict[str, Any]`   | Reserved for future extensibility                    |

Both models are frozen Pydantic models, so cases validate themselves on
construction.

## Current Case Set

Eight cases, each covering one of the requested presentation categories:

| case_id                        | Category                  | Title                                        |
| ------------------------------ | ------------------------- | -------------------------------------------- |
| `burnout_001`                  | Burnout                   | Healthcare professional burnout              |
| `anxiety_002`                  | Anxiety                   | Generalised anxiety with worry and rumination |
| `depression_like_001`          | Depression-like (no dx)   | Depression-like symptoms without diagnosis   |
| `relationship_001`             | Relationship difficulties | Relationship difficulties and attachment     |
| `academic_stress_001`          | Academic stress           | Academic stress and exam anxiety             |
| `workplace_stress_001`         | Workplace stress          | Workplace stress and interpersonal conflict  |
| `emotional_dysregulation_001`  | Emotional dysregulation   | Emotional dysregulation and intense reactions|
| `perfectionism_001`            | Perfectionism             | Perfectionism and self-worth tied to achievement |

The categories are stored in `categories` only for grouping and filtering.
Scoring never depends on them.

## Adding a New Benchmark Case

1. Construct a `BenchmarkCase` (including its `FrameworkGroundTruth`).
2. Register it:

```python
from clinical.framework_evaluation import BenchmarkCase, FrameworkGroundTruth, register_case

register_case(BenchmarkCase(
    case_id="grief_001",
    title="Prolonged grief symptoms",
    clinical_text="...",
    categories=["grief"],
    ground_truth=FrameworkGroundTruth(
        expected_topics=[...],
        expected_evidence_topics=[...],
        expected_citations=[...],
        unsafe_phrases=[...],
        differentiation_notes={...},
    ),
))
```

That's it — `evaluate_frameworks()` automatically picks up all registered
cases via `get_all_cases()`.

## Guidance for Authoring Cases

- **No diagnosis in `unsafe_phrases` overlap**: cases like
  `depression_like_001` deliberately avoid diagnostic language so the
  harness verifies the frameworks do not overstep into diagnosis.
- **Concrete `expected_citations`**: keep them short and realistic
  (e.g. `"Borkovec et al. (2004)"`) so inline-citation matching is reliable.
- **Per-framework `differentiation_notes`**: describe the lens each framework
  is expected to adopt; these are human-readable context for the diversity
  findings, not scored programmatically. Every case includes notes for CBT,
  ACT, DBT, and Humanistic (DBT emphasises emotion regulation, distress
  tolerance, mindfulness, interpersonal effectiveness, behavioural chain
  analysis, and validation).
- **Realistic `clinical_text`**: at least a few sentences of context — the
  frameworks receive this text directly as `raw_text`.
