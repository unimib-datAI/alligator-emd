---
id: configuration
title: Configuration Reference
sidebar_position: 5
---

# Configuration Reference

All configuration is handled through `AlligatorConfig`, composed of six sub-configuration groups. Every field can be passed directly to the `Alligator` constructor.

## Data Configuration

Controls input/output data handling.

| Field | Type | Default | Description |
|---|---|---|---|
| `input_csv` | `str \| Path \| DataFrame` | **required** | Path to input CSV or a pandas DataFrame |
| `output_csv` | `str` | auto | Output CSV path (derived from input filename) |
| `dataset_name` | `str` | UUID hex | Auto-generated dataset identifier |
| `table_name` | `str` | filename stem | Auto-derived from input filename |
| `target_rows` | `list[int]` | `[]` | Row indices to process (empty = all rows) |
| `target_columns` | `ColType \| None` | `None` | Dict with `NE`/`LIT`/`IGNORED` column assignments |
| `column_types` | `dict` | `{}` | Map `col_idx` → `[QID, ...]` for type-constrained search |
| `save_output` | `bool` | `False` | Enable result persistence to MongoDB |
| `save_output_to_csv` | `bool` | `False` | Write results back to a CSV file |
| `correct_qids` | `dict` | `{}` | Ground-truth QIDs for evaluation purposes |
| `dry_run` | `bool` | `False` | Skip actual processing (for testing) |
| `candidate_retrieval_only` | `bool` | `False` | Stop after Phase 2 (skip ML ranking) |
| `csv_separator` | `str` | `","` | CSV column separator character |
| `csv_header` | `str` | `"infer"` | CSV header row handling |

## Worker Configuration

Controls parallel retrieval workers.

| Field | Type | Default | Description |
|---|---|---|---|
| `num_workers` | `int` | `cpu_count // 2` | Number of parallel retrieval workers |
| `worker_batch_size` | `int` | `64` | Number of rows per worker batch |

## Retrieval Configuration

Controls LAMAPI endpoint connections.

| Field | Type | Default | Description |
|---|---|---|---|
| `entity_retrieval_endpoint` | `str` | `$ENTITY_RETRIEVAL_ENDPOINT` | Entity lookup API URL |
| `entity_retrieval_token` | `str` | `$ENTITY_RETRIEVAL_TOKEN` | Auth token for the retrieval API |
| `object_retrieval_endpoint` | `str` | `$OBJECT_RETRIEVAL_ENDPOINT` | Object relationship endpoint URL |
| `literal_retrieval_endpoint` | `str` | `$LITERAL_RETRIEVAL_ENDPOINT` | Literal values endpoint URL |
| `candidate_retrieval_limit` | `int` | `20` | Max candidates fetched per entity |
| `max_candidates_in_result` | `int` | `5` | Max candidates kept in the final output |
| `http_session_limit` | `int` | `32` | Max concurrent HTTP connections per worker |
| `http_session_ssl_verify` | `bool` | `False` | Verify SSL certificates for API calls |

## ML Configuration

Controls the machine learning ranking pipeline.

| Field | Type | Default | Description |
|---|---|---|---|
| `ranker_model_path` | `str` | `alligator/models/default.h5` | Path to the Keras ranking model |
| `reranker_model_path` | `str` | same as ranker | Path to the Keras reranking model |
| `num_ml_workers` | `int` | `2` | Number of ML worker processes |
| `ml_worker_batch_size` | `int` | `256` | Document batch size for ML prediction |
| `ml_processor_id` | `str` | `"ml-processor"` | Identifier prefix for atomic batch claiming |
| `selected_features` | `list[str]` | 27 default features | Feature names used in ML scoring |

## Feature Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `top_n_cta_cpa_freq` | `int` | `3` | Top-N type/predicate frequencies injected as global features |
| `doc_percentage_type_features` | `float` | `1.0` | Fraction of documents used to compute global type frequencies |

## Database Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `mongo_uri` | `str` | `mongodb://gator-mongodb:27017/` | MongoDB connection URI |
| `db_name` | `str` | `"alligator_db"` | Database name |
| `input_collection` | `str` | `"input_data"` | Collection for input row documents |
| `cache_collection` | `str` | `"candidate_cache"` | TTL cache for candidate API responses |
| `object_cache_collection` | `str` | `"object_cache"` | Cache for object relationship responses |
| `literal_cache_collection` | `str` | `"literal_cache"` | Cache for literal value responses |
| `error_log_collection` | `str` | `"error_logs"` | Collection for error logging |

## Match Threshold Variables

Configured via environment variables; control how ML output is translated into binary match decisions.

| Variable | Default | Description |
|---|---|---|
| `RAW_MIN_CONFIDENCE` | `0.1` | Minimum raw ML score for a cell to be eligible for auto-matching |
| `MATCH_THRESHOLD` | `0.5` | Minimum normalised score for the top candidate to be auto-matched |
| `MATCH_MARGIN_DELTA` | `0.1` | Also accept if the top candidate leads the second by at least this delta |

See [Scoring & Thresholds](scoring) for a full explanation of how these interact.
