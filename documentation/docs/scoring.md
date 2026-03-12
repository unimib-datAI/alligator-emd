---
id: scoring
title: Scoring & Thresholds
sidebar_position: 9
---

# Scoring, Normalization & Thresholds

This document explains how Alligator scores candidates, normalises them, and decides whether a cell is matched.

---

## Pipeline overview

```
ML model prediction  →  raw score  →  normalisation  →  match decision
```

The rerank stage is the final stage that produces the scores and match flags stored in MongoDB and returned by the API.

---

## 1. Raw ML score

The ML model outputs a **raw probability** in `[0, 1]` for each candidate,
representing how likely that candidate is the correct entity for the cell.

```python
raw_score = model.predict(features)[:, 1]   # class-1 probability
```

Each candidate's raw score is preserved in `raw_score` before any further
transformation so it can be recovered if needed.

---

## 2. Per-cell score normalisation

All candidates within the same cell are normalised **relative to the top
candidate in that cell**:

```python
norm_score = raw_score / max_raw_score_in_cell
```

| candidate | raw_score | norm_score |
| --------- | --------- | ---------- |
| best      | 0.80      | 1.00       |
| second    | 0.64      | 0.80       |
| third     | 0.24      | 0.30       |

**Why relative normalisation?**
The absolute raw score is model-dependent and hard to threshold universally.
Relative normalisation expresses each candidate's confidence as a _fraction of
the best available option_, making `MATCH_THRESHOLD` interpretable as
"accept if at least X% as confident as the top candidate".

---

## 3. `RAW_MIN_CONFIDENCE` — the confidence floor

**Environment variable:** `RAW_MIN_CONFIDENCE` (default `0.1`)

Before normalisation, the raw scores are checked against this floor:

```python
if max_raw_score < RAW_MIN_CONFIDENCE:
    all norm_scores = 0.0          # cell is below confidence floor
    score = raw_score              # restore raw score for display
    match = False for all          # no candidate is accepted
```

**Purpose:** guard against cells where the model is universally uncertain.
Even if a candidate would technically win the relative comparison, the model
may have very low absolute confidence for the entire cell (e.g. a vague or
ambiguous mention). When `max_raw < RAW_MIN_CONFIDENCE` the cell is treated
as unresolvable: candidates are still returned for human review but the top
candidate is **not** auto-matched.

### Tuning guide

| Value           | Effect                                                                |
| --------------- | --------------------------------------------------------------------- |
| `0.0`           | Disable the floor entirely — every cell is eligible for auto-matching |
| `0.1` (default) | Require at least 10% raw confidence before auto-matching              |
| `0.3`           | Stricter — only high-confidence cells are auto-matched                |

---

## 4. `MATCH_THRESHOLD` — the acceptance threshold

**Environment variable:** `MATCH_THRESHOLD` (default `0.5`)

Applied to the **normalised** score of the top candidate:

```python
top_auto_match = norm_score_of_top >= MATCH_THRESHOLD
```

Because the top candidate always has `norm_score = 1.0` (if above the
confidence floor), this threshold by itself would always match the top
candidate. Its real role becomes apparent when combined with `MATCH_MARGIN_DELTA`.

### Tuning guide

| Value           | Effect                                                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| `0.5` (default) | Accept if the top candidate's norm score ≥ 0.5                                                                      |
| `0.9`           | Only accept when the top candidate dominates the cell clearly                                                       |
| `1.0`           | Only accept when the top candidate is the single candidate above the floor                                          |

---

## 5. `MATCH_MARGIN_DELTA` — the margin guard

**Environment variable:** `MATCH_MARGIN_DELTA` (default `0.1`)

An _alternative_ accept condition based on how far ahead the top candidate is
from the second:

```python
top_auto_match = (norm_score_top >= MATCH_THRESHOLD)
              OR (norm_score_top - norm_score_second >= MARGIN_DELTA)
```

**Purpose:** catch cases where the top candidate has a slightly lower
normalised score (e.g. `0.45`, just below the threshold) but is clearly
better than the runner-up (e.g. second at `0.10`, margin `0.35 >> 0.1`).
Without the margin guard those cells would be left unmatched despite an
unambiguous winner.

### Tuning guide

| Value           | Effect                                                        |
| --------------- | ------------------------------------------------------------- |
| `1.0`           | Disable the margin guard — only `MATCH_THRESHOLD` decides     |
| `0.1` (default) | Accept when the top candidate leads by ≥ 10 percentage points |
| `0.5`           | Only accept when the top candidate is dramatically ahead      |

---

## 6. Match flag assignment

Only the **top candidate** can ever receive `match = True`.
All other candidates always receive `match = False`.

```python
for cand in cands_to_save:
    cand["match"] = False
cands_to_save[0]["match"] = top_auto_match   # True only if conditions above are met
```

---

## 7. Full decision flowchart

```
For each cell:
│
├─ max_raw < RAW_MIN_CONFIDENCE?
│   ├─ YES → all norm_scores = 0, restore raw scores for display
│   │         match = False for all candidates   ← STOP
│   └─ NO  → continue
│
├─ normalise: norm_score = raw_score / max_raw
│
├─ top_auto_match = (norm_top >= MATCH_THRESHOLD)
│                OR (norm_top - norm_second >= MARGIN_DELTA)
│
├─ all candidates: match = False
└─ top candidate:  match = top_auto_match
```

---

## 8. Environment variable summary

| Variable             | Default | Description                                                                                                                   |
| -------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `RAW_MIN_CONFIDENCE` | `0.1`   | Minimum raw ML score the best candidate must reach for the cell to be eligible for auto-matching                              |
| `MATCH_THRESHOLD`    | `0.5`   | Minimum normalised score the top candidate must have to be auto-matched                                                       |
| `MATCH_MARGIN_DELTA` | `0.1`   | Alternative accept condition: top candidate is auto-matched if its normalised score exceeds the second by at least this delta |

All three can be set in the Docker environment (e.g. `docker-compose.yml`) or
as regular environment variables before starting the service.
