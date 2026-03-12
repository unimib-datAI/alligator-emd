---
id: quick-start
title: Quick Start
sidebar_position: 3
---

# Quick Start

Make sure you have [installed Alligator](installation) and have MongoDB and LAMAPI running before proceeding.

## Python API

### Minimal example

Alligator will automatically classify columns and run the full pipeline:

```python
from alligator import Alligator

gator = Alligator(
    input_csv="tables/my_table.csv",
    num_workers=4,
    num_ml_workers=2,
    worker_batch_size=64,
    candidate_retrieval_limit=16,
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

### With Wikidata type constraints

Constrain candidate search per NE column to specific Wikidata entity types for higher precision:

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

## CLI

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/my_table.csv \
  --gator.num_workers 4 \
  --gator.num_ml_workers 2 \
  --gator.worker_batch_size 64 \
  --gator.candidate_retrieval_limit 16 \
  --gator.mongo_uri mongodb://localhost:27017/
```

See the full [CLI Reference](cli) for all available options.

## What Happens Next

After `run()` completes:

- Annotations are stored in MongoDB (`alligator_db` database by default)
- If `save_output_to_csv=True`, a CSV file is written alongside the input
- CEA / CTA / CPA results are accessible via the MongoDB `input_data` collection

For more details on the output format, see the [Pipeline Architecture](pipeline) page.
