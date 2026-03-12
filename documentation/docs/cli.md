---
id: cli
title: CLI Reference
sidebar_position: 4
---

# CLI Reference

The CLI is invoked via:

```bash
python3 -m alligator.cli [OPTIONS]
```

All options are auto-exposed from `Alligator.__init__` under the `--gator.*` namespace using `jsonargparse`.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--gator.input_csv` | — | **Required.** Path to the input CSV file |
| `--gator.entity_retrieval_endpoint` | `$ENTITY_RETRIEVAL_ENDPOINT` | Entity lookup API URL |
| `--gator.entity_retrieval_token` | `$ENTITY_RETRIEVAL_TOKEN` | API auth token |
| `--gator.mongo_uri` | `mongodb://gator-mongodb:27017` | MongoDB connection URI |
| `--gator.target_columns` | `None` | JSON string with `NE`/`LIT`/`IGNORED` column assignments |
| `--gator.column_types` | `{}` | JSON map of `col_idx → [QID, ...]` for type-constrained search |
| `--gator.num_workers` | `cpu_count // 2` | Number of parallel retrieval workers |
| `--gator.worker_batch_size` | `64` | Number of rows per worker batch |
| `--gator.candidate_retrieval_limit` | `20` | Max candidates fetched per entity |
| `--gator.max_candidates_in_result` | `5` | Max candidates kept in final output |
| `--gator.num_ml_workers` | `2` | Number of ML pipeline workers |
| `--gator.ml_worker_batch_size` | `256` | Batch size for ML prediction |
| `--gator.candidate_retrieval_only` | `False` | Stop after Phase 2 (skip ML ranking) |
| `--gator.save_output` | `False` | Persist output to MongoDB |
| `--gator.save_output_to_csv` | `False` | Write results to a CSV file |
| `--disable-logging` | `False` | Fully suppress all logging output |

## Examples

### Run on a CSV with defaults

```bash
python3 -m alligator.cli --gator.input_csv tables/my_table.csv
```

### With explicit column types

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/imdb_top_100.csv \
  --gator.column_types '{"0": ["Q11424"], "7": ["Q5"]}' \
  --gator.num_workers 8
```

### Manual column type assignment

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/my_table.csv \
  --gator.target_columns '{"NE": {"0": "OTHERS", "2": "LOC"}, "LIT": {"1": "NUMBER"}, "IGNORED": [3]}' \
  --gator.num_workers 4
```

### Candidate retrieval only (skip ML)

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/my_table.csv \
  --gator.candidate_retrieval_only true
```

### Save results to CSV and suppress logging

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/my_table.csv \
  --gator.save_output true \
  --gator.save_output_to_csv true \
  --disable-logging
```

## Config File

Because the CLI uses `jsonargparse`, you can also pass a YAML or JSON config file:

```bash
python3 -m alligator.cli --config my_config.yaml
```

Where `my_config.yaml` might look like:

```yaml
gator:
  input_csv: tables/my_table.csv
  num_workers: 8
  num_ml_workers: 4
  worker_batch_size: 64
  candidate_retrieval_limit: 20
  save_output: true
  save_output_to_csv: true
```
