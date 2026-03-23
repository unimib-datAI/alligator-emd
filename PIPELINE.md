# Alligator – Complete Annotation Pipeline

This document traces every class, method and database operation that runs from the moment a
client **POSTs a table** to the REST API until fully-ranked entity annotations are stored in
MongoDB and readable via `GET /datasets/{dataset}/tables/{table}`.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Phase 0 – HTTP Ingestion (FastAPI)](#phase-0--http-ingestion-fastapi)
3. [Phase 1 – Data Onboarding (`DataManager`)](#phase-1--data-onboarding-datamanager)
4. [Phase 2 – Worker Processing (`WorkerManager`)](#phase-2--worker-processing-workermanager)
   - [2a – Entity Extraction (`RowBatchProcessor._extract_entities`)](#2a--entity-extraction)
   - [2b – Candidate Retrieval (`CandidateFetcher`)](#2b--candidate-retrieval)
   - [2c – Feature Computation (`Feature.process_candidates`)](#2c--feature-computation)
   - [2d – LAMAPI Enhancement (optional)](#2d--lamapi-enhancement-optional)
   - [2e – Database Write-back](#2e--database-write-back)
5. [Phase 3 – ML Pipeline (`MLManager`)](#phase-3--ml-pipeline-mlmanager)
   - [3a – Rank Stage](#3a--rank-stage)
   - [3b – Global Frequency Computation](#3b--global-frequency-computation)
   - [3c – Rerank Stage](#3c--rerank-stage)
6. [Phase 4 – Output Generation (`OutputManager`)](#phase-4--output-generation-outputmanager)
7. [Phase 5 – API Read-back (`GET /datasets/…/tables/…`)](#phase-5--api-read-back)
8. [Class & Module Reference](#class--module-reference)
9. [MongoDB Collections Reference](#mongodb-collections-reference)

---

## High-Level Overview

```
POST /dataset/{name}/table/json
        │
        ▼
  FastAPI endpoint (alligator_api.py)
        │  background task spawned
        ▼
  Alligator.__init__          ← creates AlligatorConfig + AlligatorCoordinator
  Alligator.run()             ← delegates to AlligatorCoordinator.run()
        │
        ├─ Step 1: DataManager.onboard_data()
        │          classify columns → insert rows as MongoDB docs (status: TODO)
        │
        ├─ Step 2: WorkerManager.run_workers()
        │          N parallel processes, each running RowBatchProcessor
        │          ├─ fetch candidates from entity-retrieval endpoint
        │          ├─ compute initial features (Feature.process_candidates)
        │          ├─ optionally enrich via LAMAPI (objects + literals)
        │          └─ write candidates & update row status → DONE / rank_status: TODO
        │
        ├─ Step 3: MLManager.run_ml_pipeline()
        │          ├─ Rank stage  : MLWorker scores every candidate with a Keras model
        │          ├─ Compute global CTA/CPA frequencies (Feature.compute_global_frequencies)
        │          └─ Rerank stage: MLWorker re-scores candidates with global context features
        │
        └─ Step 4: OutputManager.save_output()
                   writes final CEA/CPA/CTA back to input_data collection (status: DONE)
```

---

## Phase 0 – HTTP Ingestion (FastAPI)

**File:** `backend/app/endpoints/alligator_api.py`  
**Endpoint:** `POST /dataset/{datasetName}/table/json`  
**Function:** `add_table(datasetName, table_upload, background_tasks, db)`

### What happens

1. The endpoint receives a `TableUpload` body (Pydantic model: `table_name`, `header`,
   `total_rows`, `classified_columns`, `data`).

2. **Dataset upsert** – `db.datasets.find_one({"dataset_name": datasetName})` checks whether
   the dataset already exists.  If not, `db.datasets.insert_one(…)` creates it.

3. **Column classification** – if `classified_columns` was not supplied (or is empty), a
   `ColumnClassifier(model_type="fast")` instance is created and
   `classify_multiple_tables([df.head(1024)])` is called to auto-detect NE / LIT / IGNORED
   columns.  The result is normalised by `format_classification(raw, header)` into the shape
   `{"NE": {col_idx: type}, "LIT": {col_idx: type}, "IGNORED": [col_idx, …]}`.

4. **Table metadata insert** – `db.tables.insert_one(table_metadata)` stores the table record
   (`status: "processing"`).

5. **Background task** – `background_tasks.add_task(run_alligator_task)` registers
   `run_alligator_task()` to execute *after* the HTTP response is sent (returns 201
   immediately).

6. Inside `run_alligator_task()`:

   ```python
   gator = Alligator(
       input_csv=pd.DataFrame(table_upload.data),
       dataset_name=datasetName,
       table_name=table_upload.table_name,
       max_candidates=3,
       entity_retrieval_endpoint=...,
       entity_retrieval_token=...,
       max_workers=8,
       candidate_retrieval_limit=30,
       model_path="./alligator/models/default.h5",
       save_output_to_csv=False,
       columns_type=classification,
   )
   gator.run()
   ```

   Once `gator.run()` returns, the same task updates:
   `db.tables.update_one(…, {"$set": {"status": "DONE", "completed_at": …}})`.

---

## Phase 1 – Data Onboarding (`DataManager`)

**File:** `alligator/manager/data.py`  
**Class:** `DataManager(DatabaseAccessMixin)`  
**Entry point:** `DataManager.onboard_data() → int`

### `Alligator.__init__` / `AlligatorConfig`

Before `onboard_data()` is ever called, `Alligator.__init__` builds an `AlligatorConfig`
object (`alligator/config.py`).  `AlligatorConfig` groups all parameters into typed
sub-configs:

| Sub-config dataclass | Key fields |
|---|---|
| `DataConfig` | `input_csv`, `dataset_name`, `table_name`, `target_columns`, `column_types` |
| `WorkerConfig` | `num_workers`, `worker_batch_size` |
| `RetrievalConfig` | `entity_retrieval_endpoint`, `entity_retrieval_token`, `candidate_retrieval_limit`, `max_candidates_in_result` |
| `MLConfig` | `ranker_model_path`, `reranker_model_path`, `num_ml_workers`, `selected_features` |
| `DatabaseConfig` | `mongo_uri`, `db_name`, `input_collection` (`input_data`), `cache_collection`, etc. |
| `FeatureConfig` | `top_n_cta_cpa_freq`, `doc_percentage_type_features` |

After config creation, `AlligatorCoordinator(config)` is instantiated.  The coordinator
immediately initialises all four managers and the shared `Feature` object.

### `DataManager.onboard_data()`

```
onboard_data()
│
├─ _classify_columns(sample)
│    ColumnClassifier(model_type="accurate")
│    .classify_multiple_tables([sample])      ← ML-based column type detection
│    Returns: {col_idx: "PERSON"|"LOCATION"|"NUMBER"|"DATE"|…}
│
├─ _process_column_types(sample, classified_columns)
│    Splits results into:
│      ne_cols   = {col_idx: NE_type}   (PERSON, LOCATION, ORGANIZATION, OTHER)
│      lit_cols  = {col_idx: LIT_type}  (NUMBER, DATE/DATETIME, STRING)
│      ignored_cols = [col_idx, …]
│      context_cols = sorted list of all non-ignored columns
│    If target_columns were passed via config, they override auto-classification.
│
└─ _process_data_chunks(…)
     Iterates over the DataFrame in chunks (4 096 rows for tables < 10 k rows).
     For each row, calls _build_document(row, …) and accumulates bulk inserts.
     Writes to MongoDB: input_collection.bulk_write([InsertOne(doc), …])
```

Each inserted MongoDB document has this shape:

```json
{
  "_id": ObjectId,
  "dataset_name": "my_dataset",
  "table_name": "my_table",
  "row_id": 0,
  "data": ["cell0", "cell1", …],
  "classified_columns": {
    "NE":  {"0": "PERSON", "3": "LOCATION"},
    "LIT": {"1": "NUMBER", "2": "DATETIME"}
  },
  "context_columns": ["0", "1", "2", "3"],
  "correct_qids": {},
  "status": "TODO",
  "rank_status": "TODO",
  "rerank_status": "TODO"
}
```

`MongoWrapper.create_indexes()` ensures compound indexes on
`(dataset_name, table_name, row_id)` for efficient retrieval.

---

## Phase 2 – Worker Processing (`WorkerManager`)

**File:** `alligator/manager/worker.py`  
**Class:** `WorkerManager(DatabaseAccessMixin)`  
**Entry point:** `WorkerManager.run_workers(feature: Feature) → None`

### Process spawning

`run_workers` counts documents with `status: "TODO"` via
`MongoWrapper.count_documents(input_collection, {"status": "TODO"})`.

It then spawns `num_workers` Python **multiprocessing** processes, each running
`WorkerManager._worker(rank, feature)` — which in turn calls
`asyncio.run(WorkerManager._worker_async(rank, feature))`.

### Per-worker async setup (`_worker_async`)

Each worker:

1. Calls `initialize_async_components()` → creates an **`aiohttp.ClientSession`** with
   `aiohttp.TCPConnector(limit=http_session_limit)`.

2. Calls `create_fetchers_and_processor(session, feature)`:
   - `CandidateFetcher(endpoint, token, limit, session, …)` — always created.
   - `ObjectFetcher(object_endpoint, token, session, …)` — created only when
     `object_retrieval_endpoint` is configured.
   - `LiteralFetcher(literal_endpoint, token, session, …)` — created only when
     `literal_retrieval_endpoint` is configured.
   - `RowBatchProcessor(dataset_name, table_name, candidate_fetcher, feature,
     object_fetcher, literal_fetcher, max_candidates_in_result, …)`

3. Divides the total document count evenly among workers using `skip` / `limit` on the
   MongoDB cursor.

4. For each document in its slice, marks the document `status: "DOING"` immediately (to
   prevent double-processing) and accumulates into batches of `worker_batch_size` (default 20).

5. For each batch: `await WorkerManager._process_batch(docs, row_processor)` →
   `await row_processor.process_rows_batch(docs)`.

---

### 2a – Entity Extraction

**Class:** `RowBatchProcessor` (`alligator/processors.py`)  
**Method:** `RowBatchProcessor._extract_entities(docs) → (List[Entity], List[RowData])`

For every document in the batch:

- Constructs a `RowData` namedtuple with `doc_id`, `row` (list of cleaned strings),
  `ne_columns`, `lit_columns`, `context_columns`, `correct_qids`, `row_index`.
- For every NE column in the row, constructs an `Entity(value, row_index, col_index,
  correct_qids, fuzzy=False, ner_type)`.
  - `_map_ner_type(raw_ner)` normalises NER labels:
    LOCATION/GPE → `"LOC"`, ORGANIZATION → `"ORG"`, PERSON → `"PERS"`, else `"OTHERS"`.
- Returns a flat `List[Entity]` (all NE cells across all rows) and the `List[RowData]`.

---

### 2b – Candidate Retrieval

**Class:** `CandidateFetcher` (`alligator/fetchers.py`)  
**Method:** `CandidateFetcher.fetch_candidates_batch(entities, fuzzies, qids, types, ner_types)`

#### Cache check

`fetch_candidates_batch_async` first de-duplicates entities: it groups by
`(value, fuzzy, qids_tuple, types_tuple, ner_type)` to avoid fetching the same mention twice.

For each unique mention, it computes a **SHA-256 cache key** via `get_cache_key(**params)` and
checks `MongoCache.get(key)`.  Cached results are returned immediately; uncached entries are
queued in `to_fetch`.

#### HTTP fetch (`_fetch_candidates`)

For each uncached entity, an async GET is issued:

```
{entity_retrieval_endpoint}?name={encoded_name}
    &limit={num_candidates}
    &fuzzy={fuzzy}
    &token={token}
    &kind=entity
    [&ids={correct_qids}]
    [&types={type_qids}]
    [&NERtype=PER|LOC]
```

The endpoint returns a list of candidate dicts:

```json
[
  {
    "id": "Q12345",
    "name": "Inception",
    "description": "2010 film by Nolan",
    "NERtype": "OTHERS",
    "types": [{"id": "Q11424"}],
    "predicates": {"0": {"P57": 1}},
    "features": {"popularity": 0.87, "es_score": 0.95, …}
  },
  …
]
```

- NER filtering: if `ner_type == "ORG"`, only candidates whose `NERtype ∈ {LOC, ORG}` are kept.
- Missing `correct_qids` are injected as placeholder dicts (`"is_placeholder": true`).
- The result is stored in `MongoCache.put(key, candidates)` (TTL 2 h, capped at 500 MB).
- On failure, the fetcher retries up to 5 times with exponential back-off (1 → 16 s).

#### Fuzzy retry

Back in `RowBatchProcessor._fetch_all_candidates`, if `fuzzy_retry=True` and any mention
returned 0 candidates, those mentions are re-fetched with `fuzzy=True`.

#### Candidate → `Candidate` objects

Raw dicts are converted to typed `Candidate` objects via `Candidate.from_dict(d)`.
Fields present in `feature.selected_features` are moved into `candidate.features` dict;
all others remain as top-level attributes (`id`, `name`, `description`, `types`,
`predicates`, …).

---

### 2c – Feature Computation

**Class:** `Feature` (`alligator/feature.py`)  
**Method:** `Feature.process_candidates(candidates: List[Candidate], row: str) → None`

Called inside `RowBatchProcessor._compute_features(row_value, mention_candidates)` for every
NE cell in the batch.

For each `Candidate` in the list, computes/preserves the following features
(in-place into `candidate.features`):

| Feature | Description |
|---|---|
| `ambiguity_mention` | How many distinct entities share the same label (from API response) |
| `corrects_tokens` | Token overlap between mention and candidate name |
| `ntoken_mention` / `ntoken_entity` | Token count of mention / entity name |
| `length_mention` / `length_entity` | Character length |
| `popularity` | Log-normalised Wikidata sitelink count |
| `pos_score` | Position score |
| `es_score` | Elasticsearch BM25-like score from retrieval endpoint |
| `ed_score` | Edit distance between mention and candidate name |
| `jaccard_score` | Token-based Jaccard similarity |
| `jaccardNgram_score` | Char-trigram Jaccard similarity |
| `p_subj_ne` | Fraction of NE columns matching this entity's subject predicates |
| `p_subj_lit_*` | Literal row/column alignment scores |
| `p_obj_ne` | Object NE overlap |
| `desc` / `descNgram` | String + n-gram similarity between candidate description and full row text (computed here via `compute_similarity_between_string`) |
| `cta_t1…t5` | Global type-frequency features — **filled in Phase 3** |
| `cpa_t1…t5` | Global predicate-frequency features — **filled in Phase 3** |

---

### 2d – LAMAPI Enhancement (optional)

If `ObjectFetcher` and `LiteralFetcher` are both configured,
`RowBatchProcessor._enhance_with_lamapi_features(row_data, entity_ids, candidates_by_col)` is
called:

1. `ObjectFetcher.fetch_objects(entity_ids)` — POST `{object_endpoint}?token=…` with
   `{"json": [qid, …]}`.  Returns entity → object predicates map.  Results cached in
   `object_cache` collection.
2. `LiteralFetcher.fetch_literals(entity_ids)` — POST `{literal_endpoint}?token=…`.  Returns
   entity → literal values map.  Results cached in `literal_cache` collection.
3. `Feature.compute_entity_entity_relationships(candidates_by_col, objects_data)` — updates
   `p_subj_ne` / `p_obj_ne` features based on predicate overlap between NE columns.
4. `Feature.compute_entity_literal_relationships(candidates_by_col, lit_columns, row, literals_data)`
   — updates `p_subj_lit_*` features by comparing candidate literal property values against
   LIT cell values in the same row.

---

### 2e – Database Write-back

After all rows in the batch are processed, `RowBatchProcessor._process_rows` issues two
`bulk_write` operations (in batches of 8 192):

**`input_data` collection** — `UpdateOne` per row:
```json
{ "$set": { "status": "DONE", "rank_status": "TODO", "rerank_status": "TODO" } }
```

**`candidates` collection** — `UpdateOne` with upsert per (row_id, col_id) pair:
```json
{
  "row_id": "42",
  "col_id": "0",
  "owner_id": ObjectId("…"),
  "candidates": [ { "id": "Q12345", "name": "…", "features": {…} }, … ]
}
```

---

## Phase 3 – ML Pipeline (`MLManager`)

**File:** `alligator/manager/ml.py`  
**Class:** `MLManager`  
**Entry point:** `MLManager.run_ml_pipeline(feature: Feature) → None`

---

### Why two stages instead of one?

After Phase 2 every NE cell has a list of candidate entities, each decorated with
~27 features computed in isolation — string similarity, token counts, popularity, BM25
score, etc.  These features are purely **local**: they compare one cell's text against one
entity in the knowledge graph without any awareness of what neighbouring cells or other rows
say about the same column.

The two-stage design mirrors how a human annotator would work:

1. **Rank** — *"Which entity best matches this text, ignoring context?"*  
   Uses only the local, row-level features that are already available.  Because global
   context has not been computed yet, the `cta_t*` / `cpa_t*` feature slots are filled
   with `0.0` as placeholders.  The goal is to produce a *provisional ranking* that is good
   enough to compute the global statistics in step 3b.

2. **Compute global statistics** — *"Across all rows, which Wikidata types dominate each
   column? Which predicates link columns together?"*  
   This is only possible once all rows have at least a provisional rank, because computing
   frequencies requires having candidate lists in place for every row.

3. **Rerank** — *"Re-score every candidate now that we know the column-level context."*  
   Injects the global type and predicate frequencies as `cta_t1…t5` / `cpa_t1…t5` features,
   then runs the Keras model a second time.  A cell whose top candidate belongs to the
   dominant type of that column gets a systematic boost; a cell that is linked to other NE
   columns via a frequent predicate also gets a boost.  Only at this point is the final CEA
   written, the candidates truncated to `max_candidates_in_result`, and the match decision
   made.

This design avoids a chicken-and-egg problem: you need ranked candidates to compute global
frequencies, and you need global frequencies to produce a good final ranking.

---

The ML pipeline uses `multiprocessing.Pool.map` with `num_ml_workers` processes for both
stages.

---

### 3a – Rank Stage

**Purpose:** produce a preliminary, local-only ranking of all candidates so that global
frequencies can be computed reliably.

Each worker runs `MLManager._ml_worker(rank, stage="rank", global_frequencies=(None,None,None))`
which instantiates:

```python
MLWorker(rank, table_name, dataset_name, stage="rank",
         model_path=ranker_model_path, batch_size=ml_worker_batch_size, …)
```

`MLWorker.run(global_frequencies)` (`alligator/ml.py`) loops until all rows are processed:

1. **Load model** — `MLWorker.load_ml_model()` calls `tensorflow.keras.models.load_model(model_path)`
   to load the `.h5` binary trained on the 27-feature vector.

2. **Claim a batch** — `find_one_and_update(rank_status="TODO" → "DOING")` atomically
   locks `batch_size` (default 256) documents, preventing duplicate processing when running
   multiple workers.  The query used is `_get_query()`:
   ```python
   {"dataset_name": …, "table_name": …, "status": "DONE", "rank_status": "TODO"}
   ```

3. **Fetch candidates** — a single `$or` query on `(row_id, owner_id)` pairs retrieves all
   candidate documents from the `candidates` collection for the whole batch at once.

4. **Build the feature matrix** — for every candidate in every NE column:
   - `cta_t1…t5` and `cpa_t1…t5` are set to `0.0` (global context not yet available).
   - `MLWorker.extract_features(cand)` reads `cand["features"][feature_name]` for each
     feature in `selected_features` (ordered list of 27 names) and returns a Python list
     of floats.
   - All candidates across all docs in the batch are stacked into a single
     `numpy.ndarray` of shape `(total_candidates, 27)`.

5. **Predict** — `model.predict(features_array, batch_size=256)` returns a `(N, 2)`
   softmax matrix; column `[:, 1]` is the match probability.

6. **Score normalisation** — scores are normalised *per cell* (not globally):
   ```
   norm_score = raw_score / max_raw_score_in_this_cell
   ```
   This ensures the top candidate always has score `1.0` and others are proportional.
   If `max_raw_score < RAW_MIN_CONFIDENCE` (env var, default `0.1`), all norms are set to
   `0.0`, signalling that the cell has no confident match.

7. **Sort and persist** — candidates are sorted by descending score.  In the rank stage
   **all** candidates are kept (no truncation); the sorted list is written back to
   `candidates` collection.  `rank_status` is set to `"DONE"` on each processed document.
   No CEA is written yet.

---

### 3b – Global Frequency Computation

**Still in the main process**, after all rank-stage workers finish.

`Feature.compute_global_frequencies(docs_to_process=1.0)` runs a **MongoDB aggregation
pipeline** over all rows in the table that have `status:"DONE", rank_status:"DONE"`:

```
$match  →  $limit  →  $lookup (join candidates)  →  $addFields (reshape to candidates_by_column)  →  $project
```

The aggregation iterates the cursor and, for each document and each NE column, looks at the
**top `top_n_cta_cpa_freq` candidates** (default 3):

- **Type frequencies (`type_frequencies`)** — for each Wikidata type QID in
  `candidate["types"]`, increments `type_frequencies[col_idx][type_qid]`.  
  After the full scan, each count is divided by `n_docs` → a per-column frequency between
  `0.0` and `1.0`.  
  *Example: if 80 % of the top candidates for column 0 have type Q11424 (film),
  `type_frequencies["0"]["Q11424"] = 0.80`.*

- **Predicate frequencies (`predicate_frequencies`)** — for each predicate in
  `candidate["predicates"]`, increments `predicate_frequencies[col_idx][pred_id]` weighted
  by the predicate's value.  Normalised by `n_docs` afterwards.

- **Predicate pair frequencies (`predicate_pair_frequencies`)** — same as above but also
  keyed by the related column index, capturing *which predicates link column A to column B*.

These three frequency dicts are passed directly into the rerank stage.

---

### 3c – Rerank Stage

**Purpose:** produce the definitive ranking using both local features *and* the global
column-context features derived in step 3b.

Same worker structure as the rank stage but `stage="rerank"` and
`global_frequencies=(type_frequencies, predicate_frequencies, predicate_pair_frequencies)`
are passed in.  `_get_query()` now matches `rank_status:"DONE", rerank_status:"TODO"`.

Inside `MLWorker.apply_ml_ranking` for the rerank:

1. **Inject global features** — for every candidate:
   - `cta_counter = type_frequencies.get(col_id, Counter())`  
     The top 5 type frequencies for types this candidate holds are sorted descending and
     stored as `cta_t1` … `cta_t5`.
   - Predicate scores are computed as `cpa_counter.get(pred_id, 0.0) * pred_value` across
     all the candidate's predicates; top 5 are stored as `cpa_t1` … `cpa_t5`.

2. **Re-score** — the feature matrix (now with non-zero `cta_*` / `cpa_*`) is fed into the
   model again (`reranker_model_path`; can be the same file as the ranker or a separately
   trained model).

3. **Score normalisation** — same relative-to-max logic as the rank stage.

4. **Truncation** — candidates are sorted descending; if any have `score > 0` only those are
   kept; otherwise the raw scores are restored.  The list is truncated to
   `max_candidates_in_result`.

5. **Match decision** — for the **top candidate only**, `match=True` is set when:
   ```
   top_score >= MATCH_THRESHOLD   (env var, default 0.4)
   OR
   (top_score − second_score) >= MATCH_MARGIN_DELTA   (env var, default 0.9)
   ```
   All other candidates unconditionally get `match=False`.  This two-condition rule avoids
   false positives when the best candidate is mediocre but has no real competition, and also
   handles cases where one candidate dominates clearly even below the threshold.

6. **Write final annotations** — `rerank_status: "DONE"` is set and three fields are written
   to the `input_data` document:
   - `cea` — `{col_id: [{id, name, description, score, match, types}, …], …}` (truncated
     to `max_candidates_in_result`)
   - `cta` — `{col_id: [top_type_qid, …]}` — the most frequent Wikidata type per column,
     derived from `keys_with_max_count(type_frequencies[col_id])`.
   - `cpa` — `{col_id: {rel_col_id: [{id, score}, …]}}` — the most frequent predicates
     linking each column pair.

---

## Phase 4 – Output Generation (`OutputManager`)

**File:** `alligator/manager/output.py`  
**Class:** `OutputManager(DatabaseAccessMixin)`  
**Entry point:** `OutputManager.save_output() → List[Dict]`

When `save_output=False` (i.e. the backend's `run_alligator_task` does not need a CSV),
this stage simply returns `[{}]`.

For CSV output mode, `document_generator(header)` cursors through `input_data` in batches of
512, calling `_extract_row_data(doc, header)` per row which synthesises columns:

```
original_value_col  | <col>_id | <col>_name | <col>_description | <col>_score
```

---

## Phase 5 – API Read-back

**Endpoint:** `GET /datasets/{dataset_name}/tables/{table_name}`  
**Function:** `get_table(…)` → `build_table_response(raw_rows, table, dataset_name, header)`

`build_table_response` (`backend/app/endpoints/response_formatter.py`) assembles the API
response from the raw MongoDB documents returned by `alligator_db.input_data.find(…)`:

| Response field | Source |
|---|---|
| `rows[].linked_entities` | Built from `doc["cea"]` — top entity per NE column |
| `semanticAnnotations.cea` | Full `doc["cea"]` with match flag derived from score threshold |
| `semanticAnnotations.cpa` | Aggregated from `doc["cpa"]`, predicate names resolved via Wikidata API |
| `semanticAnnotations.cta` | Aggregated from `doc["cta"]` |
| `metadata.column` | Per-column tag (SUBJ/NE/LIT) from `table["classified_columns"]` |

---

## Class & Module Reference

| Class | Module | Role |
|---|---|---|
| `Alligator` | `alligator/alligator.py` | Public facade; creates config & coordinator |
| `AlligatorConfig` | `alligator/config.py` | Typed, validated config container |
| `AlligatorCoordinator` | `alligator/coordinator.py` | Orchestrates the 4-phase pipeline |
| `DataManager` | `alligator/manager/data.py` | Column classification + MongoDB ingestion |
| `WorkerManager` | `alligator/manager/worker.py` | Spawns async worker processes |
| `MLManager` | `alligator/manager/ml.py` | Rank + rerank ML stages |
| `OutputManager` | `alligator/manager/output.py` | CSV / in-memory output |
| `RowBatchProcessor` | `alligator/processors.py` | Per-batch entity extraction, scoring, DB write |
| `CandidateFetcher` | `alligator/fetchers.py` | Async candidate retrieval + cache |
| `ObjectFetcher` | `alligator/fetchers.py` | LAMAPI object/predicate retrieval |
| `LiteralFetcher` | `alligator/fetchers.py` | LAMAPI literal value retrieval |
| `Feature` | `alligator/feature.py` | String/token/NER feature computation; global freq aggregation |
| `MLWorker` | `alligator/ml.py` | Loads Keras model, batches predictions, writes CEA |
| `MongoWrapper` | `alligator/mongo.py` | Thin MongoDB helper (indexes, bulk ops, error log) |
| `MongoCache` | `alligator/mongo.py` | TTL-capped MongoDB key-value cache |

---

## MongoDB Collections Reference

All collections live in the database configured by `db_name` (default: `alligator_db`).

| Collection | Written by | Read by | Key fields |
|---|---|---|---|
| `input_data` | `DataManager`, `RowBatchProcessor`, `MLWorker` | `WorkerManager`, `MLWorker`, `OutputManager`, API | `dataset_name`, `table_name`, `row_id`, `data`, `classified_columns`, `status`, `rank_status`, `rerank_status`, `cea`, `cta`, `cpa` |
| `candidates` | `RowBatchProcessor` | `MLWorker` | `row_id`, `col_id`, `owner_id`, `candidates[]` |
| `candidate_cache` | `CandidateFetcher` | `CandidateFetcher` | `key` (SHA-256), `value`, `createdAt` (TTL 2 h) |
| `object_cache` | `ObjectFetcher` | `ObjectFetcher` | `key` (entity QID), `value` |
| `literal_cache` | `LiteralFetcher` | `LiteralFetcher` | `key` (entity QID), `value` |
| `error_logs` | `MongoWrapper.log_to_db` | — | `level`, `message`, `traceback`, `timestamp` |

The **backend** database (`alligator_backend_db`) holds two additional collections managed
exclusively by the FastAPI layer:

| Collection | Key fields |
|---|---|
| `datasets` | `dataset_name`, `total_tables`, `total_rows`, `created_at` |
| `tables` | `dataset_name`, `table_name`, `header`, `total_rows`, `status`, `classified_columns`, `created_at`, `completed_at` |
