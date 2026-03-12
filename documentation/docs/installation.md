---
id: installation
title: Installation
sidebar_position: 2
---

# Installation

## Requirements

- Python ≥ 3.9
- MongoDB (local or via Docker)
- A running **LAMAPI** entity retrieval service

## Install from Source

```bash
git clone https://github.com/unimib-datAI/alligator-emd.git
cd alligator-emd
pip install -e .
```

### Optional Extras

```bash
# FastAPI REST backend
pip install -e ".[app]"

# Development tools (pytest, black, mypy, etc.)
pip install -e ".[dev]"
```

## Docker Compose

The repository ships a ready-to-use `docker-compose.yml` that starts both the Alligator backend and MongoDB:

```bash
docker compose up
```

For debug mode with Node.js inspector:

```bash
docker compose -f docker-compose.debug.yml up
```

## Environment Variables

Create a `.env` file in the project root (a sample is provided):

| Variable | Description |
|---|---|
| `ENTITY_RETRIEVAL_ENDPOINT` | LAMAPI entity lookup endpoint URL |
| `ENTITY_RETRIEVAL_TOKEN` | Auth token for the retrieval API |
| `OBJECT_RETRIEVAL_ENDPOINT` | Object relationship endpoint URL |
| `LITERAL_RETRIEVAL_ENDPOINT` | Literal values endpoint URL |
| `MONGO_URI` | MongoDB connection URI (default: `mongodb://localhost:27017/`) |

### Match Threshold Variables

These control the ML match decision behaviour (see [Scoring & Thresholds](scoring)):

| Variable | Default | Description |
|---|---|---|
| `RAW_MIN_CONFIDENCE` | `0.1` | Minimum raw ML confidence required for auto-matching |
| `MATCH_THRESHOLD` | `0.5` | Minimum normalised score for the top candidate to be accepted |
| `MATCH_MARGIN_DELTA` | `0.1` | Accept if top candidate leads the second candidate by at least this margin |
