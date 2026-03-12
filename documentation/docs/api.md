---
id: api
title: Python API Reference
sidebar_position: 7
---

# Python API Reference

## `Alligator`

The primary entry point for programmatic use.

```python
from alligator import Alligator

gator = Alligator(input_csv="tables/my_table.csv", **kwargs)
gator.run()
```

### Constructor Parameters

All `AlligatorConfig` fields can be passed as keyword arguments. Commonly used ones:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_csv` | `str \| Path \| DataFrame` | **required** | Input data source |
| `num_workers` | `int` | `cpu_count // 2` | Parallel retrieval workers |
| `worker_batch_size` | `int` | `64` | Rows per worker batch |
| `candidate_retrieval_limit` | `int` | `20` | Max candidates fetched per entity |
| `max_candidates_in_result` | `int` | `5` | Max candidates in output |
| `num_ml_workers` | `int` | `2` | ML pipeline workers |
| `ml_worker_batch_size` | `int` | `256` | ML batch size |
| `mongo_uri` | `str` | `mongodb://gator-mongodb:27017/` | MongoDB URI |
| `target_columns` | `ColType \| None` | `None` | Manual column type overrides |
| `column_types` | `dict` | `{}` | Wikidata type constraints per column |
| `candidate_retrieval_only` | `bool` | `False` | Stop after Phase 2 |
| `save_output` | `bool` | `False` | Persist results to MongoDB |
| `save_output_to_csv` | `bool` | `False` | Write results to CSV |

See [Configuration Reference](configuration) for the full parameter list.

### Methods

#### `run() → None`

Runs the complete pipeline: onboard data → worker processing → ML ranking → output.

```python
gator.run()
```

#### `onboard_data() → None`

Runs only Phase 1: column classification and MongoDB ingestion.

```python
gator.onboard_data()
```

#### `save_output() → None`

Manually triggers Phase 4: output assembly and optional CSV write.

```python
gator.save_output()
```

#### `close_mongo_connection() → None`

Closes the MongoDB connection pool. Call this when done if managing the lifecycle manually.

```python
gator.close_mongo_connection()
```

---

## Key Types

### `ColType`

Used for manual column type assignment via `target_columns`.

```python
from alligator.types import ColType

target_columns: ColType = {
    "NE":      {0: "OTHERS", 2: "LOC"},   # col_idx → NER label
    "LIT":     {1: "NUMBER", 3: "STRING"}, # col_idx → literal type
    "IGNORED": [4, 5],
}
```

**NER Labels:** `"LOC"`, `"ORG"`, `"PERS"`, `"OTHERS"`

**Literal Types:** `"NUMBER"`, `"STRING"`, `"DATE"`, `"BOOLEAN"`

### `Entity`

Represents a named entity extracted from an NE cell.

```python
@dataclass
class Entity:
    value: str          # cell text value
    row_index: int
    col_index: int
    correct_qids: list[str]   # ground-truth QIDs (for evaluation)
    fuzzy: bool               # whether fuzzy retrieval was used
    ner_type: str             # NER label
```

### `Candidate`

Represents a Wikidata entity candidate for a cell.

```python
@dataclass
class Candidate:
    id: str             # Wikidata QID (e.g. "Q12345")
    name: str           # entity label
    description: str    # entity description
    score: float        # normalised ML score [0, 1]
    features: dict      # 27-feature vector
    types: list[str]    # Wikidata type QIDs
    predicates: dict    # predicate map for CPA
    matches: bool       # True for the auto-matched candidate

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> 'Candidate': ...
```

---

## Feature Names

The 27 default features used in ML scoring are accessible via:

```python
from alligator.feature import DEFAULT_FEATURES

print(DEFAULT_FEATURES)
# ['ed_score', 'jaccard', 'jaro_winkler', ..., 'cta_t1', 'cpa_t1', ...]
```

Feature categories:
- **String similarity**: exact match, Levenshtein, Jaro-Winkler, Jaccard
- **N-gram overlap**: character and token level
- **Description similarity**: `desc`, `descNgram`
- **Type indicators**: `ntype_LOC`, `ntype_ORG`, `ntype_PERS`, `ntype_OTHERS`
- **Column relationship features** (rerank only): `cta_t1…t5`, `cpa_t1…t5`, `lit_*`
- **Retrieval score**: `ed_score` (score from LAMAPI)
