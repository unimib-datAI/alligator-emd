---
id: examples
title: Examples
sidebar_position: 8
---

# Examples

## Basic Usage

```python
from alligator import Alligator

gator = Alligator(
    input_csv="tables/imdb_top_100.csv",
    num_workers=4,
    num_ml_workers=2,
    worker_batch_size=64,
    candidate_retrieval_limit=16,
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

## With Wikidata Type Constraints

Constrain candidate search per NE column to specific Wikidata entity types. This significantly improves precision when you know the expected entity types in each column.

```python
from alligator import Alligator

# Map column index (as string) → list of Wikidata QIDs
column_types = {
    "0": ["Q11424"],   # film
    "3": ["Q483394"],  # music genre
    "7": ["Q5"],       # human
    "8": ["Q5"],       # human
}

gator = Alligator(
    input_csv="tables/imdb_top_100.csv",
    column_types=column_types,
    num_workers=4,
    num_ml_workers=2,
    worker_batch_size=64,
    candidate_retrieval_limit=16,
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

## Manual Column Type Assignment

Override automatic column classification with explicit NE/LIT/IGNORED assignments:

```python
from alligator import Alligator
from alligator.types import ColType

target_columns: ColType = {
    "NE":      {0: "OTHERS", 2: "LOC"},
    "LIT":     {1: "NUMBER", 3: "STRING"},
    "IGNORED": [4, 5],
}

gator = Alligator(
    input_csv="tables/my_table.csv",
    target_columns=target_columns,
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

## Candidate Retrieval Only

Stop the pipeline after Phase 2 to inspect raw candidates before committing to ML scoring:

```python
gator = Alligator(
    input_csv="tables/my_table.csv",
    candidate_retrieval_only=True,
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

## Process a Subset of Rows

```python
gator = Alligator(
    input_csv="tables/my_table.csv",
    target_rows=[0, 1, 2, 10, 11],
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

## Save Results to CSV

```python
gator = Alligator(
    input_csv="tables/my_table.csv",
    save_output=True,
    save_output_to_csv=True,
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

The output CSV will be written to the same directory as the input file.

## CLI Examples

### Basic run

```bash
python3 -m alligator.cli --gator.input_csv tables/my_table.csv
```

### With column types and increased parallelism

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/imdb_top_100.csv \
  --gator.column_types '{"0": ["Q11424"], "7": ["Q5"]}' \
  --gator.num_workers 8 \
  --gator.num_ml_workers 4
```

### Using a YAML config file

```yaml title="config.yaml"
gator:
  input_csv: tables/my_table.csv
  num_workers: 8
  num_ml_workers: 4
  worker_batch_size: 64
  candidate_retrieval_limit: 20
  save_output: true
  save_output_to_csv: true
```

```bash
python3 -m alligator.cli --config config.yaml
```
