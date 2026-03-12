---
id: intro
title: Introduction
sidebar_position: 1
slug: /intro
---

# Alligator

**Alligator** is a Python library for **entity linking over tabular data**. Given a CSV table, it automatically:

1. Identifies **Named Entity (NE)** columns (or accepts manual assignments)
2. Fetches candidate Wikidata entities for each cell via the **LAMAPI** retrieval service
3. Computes ~27 similarity and overlap **features** per candidate
4. Runs a two-stage Keras ML pipeline (**rank → rerank**) to score and rank candidates
5. Produces three types of SemTab-compatible annotations:
   - **CEA** — Cell Entity Annotation: links each NE cell to a Wikidata QID
   - **CTA** — Column Type Annotation: infers the Wikidata type for each NE column
   - **CPA** — Column Property Annotation: infers relationships between NE columns

Results are stored in **MongoDB** and can optionally be written back to CSV.

## Key Features

- Automatic column type classification (NE / LIT / IGNORED) via `column-classifier`
- SHA-256 keyed MongoDB TTL cache for API responses (avoids redundant lookups)
- Fuzzy-retry candidate retrieval when exact match returns no results
- 5-attempt exponential backoff on retrieval failures
- Two-stage ML ranking: rank (local features) → rerank (global CTA/CPA frequency features)
- Parallel multiprocessing workers for scalable throughput
- FastAPI REST backend for integration into larger systems
- Docker Compose setup for easy deployment

## Architecture Overview

```
CSV / DataFrame
      │
      ▼
DataManager       — column classification, MongoDB onboarding
      │
      ▼
WorkerManager     — N async workers: entity extraction → candidate fetch → feature computation
      │
      ▼
MLManager         — rank → compute global frequencies → rerank
      │
      ▼
OutputManager     — CSV output + MongoDB annotations
```

## Quick Links

- [Installation](installation) — how to install and configure Alligator
- [Quick Start](quick-start) — run your first table annotation in minutes
- [CLI Reference](cli) — full command-line options
- [Configuration Reference](configuration) — all parameters explained
- [Pipeline Architecture](pipeline) — deep dive into how the pipeline works
