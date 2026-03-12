---
id: custom-processor
title: Custom ML Processors
sidebar_position: 10
---

# Custom ML Processors

Alligator's ML phase is fully pluggable. Instead of the built-in `MLProcessor` (Keras two-stage ranker), you can swap in any logic you like — an LLM reranker, a rule-based scorer, a remote inference service, or anything else — without touching the rest of the pipeline.

## How the registry works

All processors are registered automatically through Python's `__init_subclass__` hook inside `BaseProcessor`:

```python
class BaseProcessor:
    registry = {}         # processor_id → class
    processor_id = None   # set this in your subclass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.processor_id:
            BaseProcessor.registry[cls.processor_id] = cls

    def process(self, data):
        raise NotImplementedError
```

Every subclass that sets `processor_id` is added to `BaseProcessor.registry` **at import time**. `MLManager` then resolves the processor by that key:

```python
processor = BaseProcessor.registry[processor_id](self.config)
processor.process(feature)
```

## Built-in processors

| `processor_id` | Class | What it does |
|---|---|---|
| `ml-processor` | `MLProcessor` | Two-stage Keras ranking (default) |
| `llm-processor` | `LLMProcessor` | OpenAI-compatible LLM reranking |

## Creating your own processor

### 1. Subclass `BaseProcessor` and set `processor_id`

```python
# my_project/my_processor.py
from alligator.manager.processors.BaseProcessor import BaseProcessor
from alligator.config import AlligatorConfig


class MyProcessor(BaseProcessor):
    processor_id = "my-processor"   # must be unique

    def __init__(self, config: AlligatorConfig):
        self.config = config

    def process(self, feature):
        """
        Called by MLManager after worker processing completes.

        Args:
            feature: alligator.feature.Feature instance.
                     Use feature.compute_global_frequencies() if you need
                     CTA/CPA type frequency data.
        """
        # your ranking / annotation logic here
        ...
```

### 2. Make it importable before calling `run()`

The processor **must be imported** before `Alligator.run()` so that `__init_subclass__` fires and the class lands in the registry. There are two ways to do this:

#### Option A — Import it yourself

```python
import my_project.my_processor   # registers MyProcessor in BaseProcessor.registry

from alligator import Alligator

gator = Alligator(
    input_csv="tables/my_table.csv",
    ml_processor_id="my-processor",   # tell Alligator which processor to use
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

#### Option B — Drop it in the processors package

Place your file inside `alligator/manager/processors/`. The package `__init__.py` auto-imports every module in that directory:

```python
# alligator/manager/processors/__init__.py
import pkgutil, importlib
for module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module.name}")
```

So any `.py` file you add there is discovered and registered automatically — no extra import needed.

### 3. Pass `processor_id` to Alligator

Either at construction time:

```python
gator = Alligator(
    input_csv="tables/my_table.csv",
    ml_processor_id="my-processor",
)
gator.run()
```

Or at call time (overrides the constructor value):

```python
gator = Alligator(input_csv="tables/my_table.csv")
gator.run(processor_id="my-processor")
```

## The `process(self, feature)` contract

Your `process` method receives an `alligator.feature.Feature` instance. It must read candidate documents from MongoDB and write `cea`/`cta`/`cpa` annotations back to the `input_data` collection before returning.

Key things available through `self.config`:

| Config field | Common use |
|---|---|
| `self.config.data.dataset_name` | Filter documents by dataset |
| `self.config.data.table_name` | Filter documents by table |
| `self.config.database.mongo_uri` | Connect to MongoDB |
| `self.config.database.db_name` | Database name |
| `self.config.database.input_collection` | Collection to read/write (`input_data`) |
| `self.config.retrieval.max_candidates_in_result` | Max candidates to keep per cell |
| `self.config.ml.ml_worker_batch_size` | Suggested batch size |

Use `feature.compute_global_frequencies()` if your processor needs global type/predicate frequency data (same as the built-in rerank stage):

```python
type_freqs, pred_freqs, pair_freqs = feature.compute_global_frequencies(
    docs_to_process=self.config.feature.doc_percentage_type_features,
    random_sample=False,
)
```

## Complete example — random-score processor

This minimal processor assigns a random score to each candidate (useful for testing the pipeline end-to-end):

```python
import random
from pymongo.operations import UpdateOne
from alligator.manager.processors.BaseProcessor import BaseProcessor
from alligator.database import DatabaseAccessMixin


class RandomProcessor(BaseProcessor, DatabaseAccessMixin):
    processor_id = "random-processor"

    def __init__(self, config):
        DatabaseAccessMixin.__init__(self)
        self.config = config
        self._mongo_uri = config.database.mongo_uri or "mongodb://localhost:27017/"
        self._db_name = config.database.db_name or "alligator_db"

    def process(self, feature):
        db = self.get_db()
        input_col = db[self.config.database.input_collection]
        cand_col = db["candidates"]

        query = {
            "dataset_name": self.config.data.dataset_name,
            "table_name": self.config.data.table_name,
            "status": "DONE",
        }

        input_updates = []
        cand_updates = []

        for doc in input_col.find(query):
            cea, cta, cpa = {}, {}, {}

            for record in cand_col.find({"owner_id": doc["_id"]}):
                col_id = str(record["col_id"])
                cands = record.get("candidates", [])

                # Score randomly and pick a winner
                for c in cands:
                    c["score"] = random.random()
                    c["match"] = False
                cands.sort(key=lambda c: c["score"], reverse=True)
                if cands:
                    cands[0]["match"] = True

                max_cands = self.config.retrieval.max_candidates_in_result
                to_save = cands[:max_cands]

                cea[col_id] = [
                    {k: v for k, v in c.items()
                     if k in {"id", "name", "score", "match", "description", "types"}}
                    for c in to_save
                ]
                if to_save and to_save[0]["match"]:
                    cta[col_id] = [t["id"] for t in to_save[0].get("types", []) if t.get("id")][:1]
                else:
                    cta[col_id] = []
                cpa[col_id] = {}

                cand_updates.append(
                    UpdateOne(
                        {"_id": record["_id"]},
                        {"$set": {"candidates": to_save}},
                    )
                )

            input_updates.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "rank_status": "DONE",
                        "rerank_status": "DONE",
                        "cea": cea,
                        "cta": cta,
                        "cpa": cpa,
                    }},
                )
            )

        if cand_updates:
            db["candidates"].bulk_write(cand_updates, ordered=False)
        if input_updates:
            input_col.bulk_write(input_updates, ordered=False)
```

Then use it:

```python
import my_project.random_processor   # registers RandomProcessor

from alligator import Alligator

gator = Alligator(
    input_csv="tables/my_table.csv",
    ml_processor_id="random-processor",
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

## The built-in LLM processor

The library ships `LLMProcessor` (`processor_id = "llm-processor"`) as a ready-to-use alternative to the Keras model. It uses any OpenAI-compatible API to select the best candidate per cell.

Configure it via environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |
| `LLM_API_KEY` | — | API key |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Model name |
| `LLM_GROUPING` | `none` | Set to `row` for one LLM call per row instead of per cell |
| `LLM_MAX_RETRIES` | `5` | Max retry attempts on failure |
| `LLM_BACKOFF_INITIAL` | `0.5` | Initial backoff seconds (doubles each retry, max 60 s) |

Usage:

```python
from alligator import Alligator

gator = Alligator(
    input_csv="tables/my_table.csv",
    ml_processor_id="llm-processor",
    mongo_uri="mongodb://localhost:27017/",
)
gator.run()
```

The LLM processor skips the two-stage Keras ranking entirely. It marks all rank-stage documents as complete in one pass, then uses the LLM to select the best candidate, writing `cea`/`cta`/`cpa` annotations just like the ML processor does.
