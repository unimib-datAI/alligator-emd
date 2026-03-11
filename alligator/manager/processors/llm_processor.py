import os
import time
import json
import re
import json
import re

try:
    import httpx
except Exception:
    httpx = None
from typing import Any, Dict, List

from openai import OpenAI
from pymongo.operations import UpdateOne

from alligator.config import AlligatorConfig
from alligator.database import DatabaseAccessMixin

from .BaseProcessor import BaseProcessor


class _LLMReranker(DatabaseAccessMixin):
    """Handles LLM-based candidate selection against MongoDB collections."""

    def __init__(self, config: AlligatorConfig, client: Any, model: str):
        super().__init__()
        self._mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        self._db_name = config.database.db_name or "alligator_db"
        self.config = config
        self.client = client
        self.model = model
        # grouping can be 'none' or 'row'
        self._llm_grouping = os.environ.get("LLM_GROUPING", "none").lower()
        self.base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        # track recent 429 events to adaptively pause
        self._last_429_time = 0.0
        self._consecutive_429s = 0

    def run_rank_passthrough(self):
        """Mark all rank=TODO docs as rank=DONE without ML scoring."""
        db = self.get_db()
        db[self.config.database.input_collection].update_many(
            {
                "dataset_name": self.config.data.dataset_name or "default_dataset",
                "table_name": self.config.data.table_name or "default_table",
                "status": "DONE",
                "rank_status": "TODO",
            },
            {"$set": {"rank_status": "DONE"}},
        )

    def run_llm_rerank(self):
        """Process all rerank=TODO documents and select best candidates via LLM."""
        db = self.get_db()
        input_col = db[self.config.database.input_collection]
        cand_col = db["candidates"]
        dataset_name = self.config.data.dataset_name or "default_dataset"
        table_name = self.config.data.table_name or "default_table"
        batch_size = self.config.ml.ml_worker_batch_size
        max_candidates = self.config.retrieval.max_candidates_in_result

        rerank_query = {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "status": "DONE",
            "rank_status": "DONE",
            "rerank_status": "TODO",
        }

        while True:
            # Claim a batch atomically
            batch_docs = []
            for _ in range(batch_size):
                doc = input_col.find_one_and_update(
                    rerank_query,
                    {"$set": {"rerank_status": "DOING"}},
                    projection={"_id": 1, "row_id": 1, "data": 1, "classified_columns": 1},
                )
                if doc is None:
                    break
                doc["candidates"] = {}
                batch_docs.append(doc)

            if not batch_docs:
                break

            # Fetch all candidates for this batch in one query
            pair_conditions = [
                {"row_id": str(doc["row_id"]), "owner_id": doc["_id"]} for doc in batch_docs
            ]
            doc_map = {(str(d["row_id"]), d["_id"]): d for d in batch_docs}
            for record in cand_col.find(
                {"$or": pair_conditions},
                projection={"_id": 0, "candidates": 1, "col_id": 1, "row_id": 1, "owner_id": 1},
            ):
                doc = doc_map.get((record.get("row_id"), record.get("owner_id")))
                if doc:
                    col_id = record.get("col_id")
                    doc["candidates"].setdefault(col_id, []).extend(record.get("candidates", []))

            cand_updates = []
            input_updates = []
            # If grouping is enabled, do one LLM call per row with all columns
            if self._llm_grouping == "row":
                for doc in batch_docs:
                    cea_results: Dict[str, Any] = {}
                    cta_results: Dict[str, Any] = {}
                    cpa_results: Dict[str, Any] = {}
                    row_data = doc.get("data", [])

                    # candidates is a dict col_id -> list
                    try:
                        row_choices = self._ask_llm_row(
                            row_data, doc.get("candidates", {}), doc.get("classified_columns", {})
                        )
                    except Exception:
                        row_choices = {}

                    for col_id, candidates in doc.get("candidates", {}).items():
                        if not candidates:
                            continue

                        chosen_id = (
                            row_choices.get(str(col_id)) or row_choices.get(col_id) or "none"
                        )

                        for cand in candidates:
                            cand["match"] = cand["id"] == chosen_id
                            cand["score"] = 1.0 if cand["match"] else 0.0

                        sorted_cands = sorted(
                            candidates, key=lambda c: c.get("score", 0.0), reverse=True
                        )
                        cands_to_save = (
                            sorted_cands[:max_candidates] if max_candidates > 0 else sorted_cands
                        )

                        cea_results[col_id] = [
                            {
                                k: v
                                for k, v in c.items()
                                if k in {"score", "id", "name", "description", "types", "match"}
                            }
                            for c in cands_to_save
                        ]

                        winning_types = (
                            [t.get("id") for t in cands_to_save[0].get("types", []) if t.get("id")]
                            if cands_to_save and cands_to_save[0].get("match")
                            else []
                        )
                        cta_results[col_id] = winning_types[:1]
                        cpa_results[col_id] = {}

                        cand_updates.append(
                            UpdateOne(
                                {
                                    "row_id": str(doc["row_id"]),
                                    "col_id": str(col_id),
                                    "owner_id": doc["_id"],
                                },
                                {"$set": {"candidates": cands_to_save}},
                            )
                        )

                    input_updates.append(
                        UpdateOne(
                            {"_id": doc["_id"]},
                            {
                                "$set": {
                                    "rerank_status": "DONE",
                                    "cea": cea_results,
                                    "cta": cta_results,
                                    "cpa": cpa_results,
                                }
                            },
                        )
                    )
            else:
                # Fallback: original per-column serial processing
                for doc in batch_docs:
                    cea_results: Dict[str, Any] = {}
                    cta_results: Dict[str, Any] = {}
                    cpa_results: Dict[str, Any] = {}
                    row_data = doc.get("data", [])

                    for col_id, candidates in doc["candidates"].items():
                        if not candidates:
                            continue

                        try:
                            cell_value = str(row_data[int(col_id)]) if row_data else ""
                        except (IndexError, ValueError):
                            cell_value = ""

                        col_name = None
                        try:
                            col_name = (doc.get("classified_columns") or {}).get(str(col_id))
                        except Exception:
                            col_name = None
                        chosen_id = self._ask_llm(
                            cell_value, row_data, candidates, col_name=col_name
                        )

                        for cand in candidates:
                            cand["match"] = cand["id"] == chosen_id
                            cand["score"] = 1.0 if cand["match"] else 0.0

                        sorted_cands = sorted(
                            candidates, key=lambda c: c.get("score", 0.0), reverse=True
                        )
                        cands_to_save = (
                            sorted_cands[:max_candidates] if max_candidates > 0 else sorted_cands
                        )

                        cea_results[col_id] = [
                            {
                                k: v
                                for k, v in c.items()
                                if k in {"score", "id", "name", "description", "types", "match"}
                            }
                            for c in cands_to_save
                        ]

                        # CTA: take the top type of the matched entity if available
                        winning_types = (
                            [t.get("id") for t in cands_to_save[0].get("types", []) if t.get("id")]
                            if cands_to_save and cands_to_save[0].get("match")
                            else []
                        )
                        cta_results[col_id] = winning_types[:1]
                        cpa_results[col_id] = {}

                        cand_updates.append(
                            UpdateOne(
                                {
                                    "row_id": str(doc["row_id"]),
                                    "col_id": str(col_id),
                                    "owner_id": doc["_id"],
                                },
                                {"$set": {"candidates": cands_to_save}},
                            )
                        )

                    input_updates.append(
                        UpdateOne(
                            {"_id": doc["_id"]},
                            {
                                "$set": {
                                    "rerank_status": "DONE",
                                    "cea": cea_results,
                                    "cta": cta_results,
                                    "cpa": cpa_results,
                                }
                            },
                        )
                    )

            bulk_size = 8192
            for i in range(0, len(cand_updates), bulk_size):
                db["candidates"].bulk_write(cand_updates[i : i + bulk_size], ordered=False)
            for i in range(0, len(input_updates), bulk_size):
                db[self.config.database.input_collection].bulk_write(
                    input_updates[i : i + bulk_size], ordered=False
                )

    def run_llm_process_once(self):
        """
        Perform a single LLM processing pass: ensure ranking status is set,
        then run the LLM reranking loop. This provides a one-call entrypoint
        that both prepares and processes documents.
        """
        # Ensure any rank TODO docs are advanced to DONE so we can rerank in one pass
        db = self.get_db()
        db[self.config.database.input_collection].update_many(
            {
                "dataset_name": self.config.data.dataset_name or "default_dataset",
                "table_name": self.config.data.table_name or "default_table",
                "status": "DONE",
                "rank_status": "TODO",
            },
            {"$set": {"rank_status": "DONE"}},
        )

        # Run the rerank loop
        self.run_llm_rerank()

    def _ask_llm(
        self,
        cell_value: str,
        row_context: List[Any],
        candidates: List[Dict],
        col_name: str | None = None,
    ) -> str:
        """
        Ask the LLM to identify the best matching Wikidata candidate for a cell.
        Returns the winning candidate ID, or 'none' if no match is appropriate.
        """
        # Include types (name + optional description) for each candidate
        cand_lines = []
        for c in candidates:
            types = c.get("types") or []
            if types:
                types_str = ", ".join(
                    f"{t.get('name') or t.get('id')}{' (' + (t.get('description') or '') + ')' if t.get('description') else ''}"
                    for t in types
                )
            else:
                types_str = ""
            cand_lines.append(
                f"- ID: {c.get('id')} | Name: {c.get('name','')} | Desc: {c.get('description','')}"
                + (f" | Types: {types_str}" if types_str else "")
            )
        candidate_lines = "\n".join(cand_lines)
        # Include column name when available to provide extra context
        col_info = f"Column name: {col_name}\n" if col_name else ""

        prompt = (
            f"You are an entity linking assistant. Given a table cell value, the column name (if any), and a list of "
            f"Wikidata candidates, identify the best matching entity.\n\n"
            f"{col_info}"
            f'Cell value: "{cell_value}"\n'
            f"Row context (all cells): {row_context}\n\n"
            f"Candidates:\n{candidate_lines}\n\n"
            f"Reply with ONLY the Wikidata ID (e.g. Q42) of the best match, "
            f"or 'none' if no candidate is a good match. Do not explain."
        )

        # Make the LLM call with limited concurrency and retries/backoff on rate limits.
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "5"))
        backoff = float(os.environ.get("LLM_BACKOFF_INITIAL", "0.5"))

        response = None
        # Make calls serially; retry/backoff on transient errors
        for attempt in range(max_retries):
            # If we've recently seen 429s, add a small proactive delay
            if self._consecutive_429s > 0 and (time.time() - self._last_429_time) < 60:
                extra = min(5 * self._consecutive_429s, 30)
                time.sleep(extra)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=16,
                    temperature=0,
                )
                # success -> reset 429 counters
                self._consecutive_429s = 0
                break
            except Exception as e:
                # Try to detect a 429 response code in the exception message or attributes
                is_429 = False
                retry_after = None
                try:
                    msg = str(e)
                    if "429" in msg or "Too Many Requests" in msg:
                        is_429 = True
                except Exception:
                    pass

                # If httpx is available and exception exposes response, try to inspect
                try:
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        status = getattr(resp, "status_code", None) or getattr(
                            resp, "status", None
                        )
                        if status == 429 or str(status) == "429":
                            is_429 = True
                            # headers may contain retry info
                            headers = getattr(resp, "headers", {}) or {}
                            retry_after = (
                                headers.get("Retry-After")
                                or headers.get("retry-after")
                                or headers.get("x-ratelimit-reset")
                            )
                except Exception:
                    pass

                if is_429:
                    self._consecutive_429s += 1
                    self._last_429_time = time.time()
                    # try to fetch OpenRouter rate-limit hints if available
                    info = {}
                    if httpx is not None:
                        try:
                            r = httpx.get(f"{self.base_url}/models", timeout=5.0)
                            info = {k.lower(): v for k, v in r.headers.items()}
                            # prefer Retry-After header
                            retry_after = (
                                retry_after
                                or info.get("retry-after")
                                or info.get("x-ratelimit-reset")
                            )
                        except Exception:
                            info = {}

                    if retry_after:
                        try:
                            wait = int(float(retry_after))
                        except Exception:
                            # try parse as epoch reset time
                            try:
                                wait = max(0, int(float(retry_after) - time.time()))
                            except Exception:
                                wait = backoff
                        time.sleep(wait)
                    else:
                        time.sleep(backoff)

                    backoff = min(backoff * 2, 60.0)
                    # continue retrying
                    continue

                # non-429 transient error: exponential backoff
                if attempt == max_retries - 1:
                    response = None
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

        # Extract text from multiple possible response shapes returned by
        # OpenAI-compatible libraries (robust against None fields).
        answer = None
        try:
            # Try attribute-style access first
            if hasattr(response, "choices") and response.choices:
                choice0 = response.choices[0]
                # common: choice0.message.content
                msg = getattr(choice0, "message", None)
                if msg:
                    answer = getattr(msg, "content", None) or (
                        msg.get("content") if isinstance(msg, dict) else None
                    )
                # older/alternative clients: choice0.text
                if not answer:
                    answer = getattr(choice0, "text", None)
                # delta streaming shape: choice0.delta.content
                if not answer:
                    delta = getattr(choice0, "delta", None)
                    if delta:
                        answer = getattr(delta, "content", None) or (
                            delta.get("content") if isinstance(delta, dict) else None
                        )

            # fallback: dict-like access
            if not answer and isinstance(response, dict):
                choices = response.get("choices") or []
                if choices:
                    c0 = choices[0]
                    if isinstance(c0, dict):
                        # try message.content, text, delta.content
                        answer = (
                            (c0.get("message") or {}).get("content")
                            or c0.get("text")
                            or (c0.get("delta") or {}).get("content")
                        )

        except Exception:
            answer = None

        if not answer:
            # Log minimal info (avoid noisy prints) and treat as no-match
            return "none"

        answer = str(answer).strip()

        # Extract a valid candidate QID from the response, tolerating extra text
        valid_ids = {c["id"] for c in candidates}
        for token in answer.split():
            token = token.strip(".,\"'()")
            if token in valid_ids:
                return token

        return "none"

    def _extract_json_from_text(self, text: str):
        """Try to extract a JSON object from text and parse it."""
        if not text:
            return None
        # Try to find the first JSON object in the text
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            # maybe it's already a simple token list
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            # try to sanitize trailing commas
            s = re.sub(r",\s*}\s*$", "}", m.group(0))
            try:
                return json.loads(s)
            except Exception:
                return None

    def _ask_llm_row(
        self,
        row_data: List[Any],
        candidates_map: Dict[str, List[Dict]],
        col_names: Dict[str, str] | None = None,
    ) -> Dict[str, str]:
        """
        Ask the LLM once for all columns in a row. Returns a mapping col_id->chosen_id.
        Falls back to empty dict on failure.
        """
        # Build candidate listing per column
        parts = [f"Row context: {row_data}", "Candidates per column:"]
        for col_id, cands in candidates_map.items():
            if not cands:
                continue
            col_name = None
            if col_names:
                col_name = col_names.get(str(col_id)) or col_names.get(col_id)
            lines = [f"Column {col_id}: {col_name}" if col_name else f"Column {col_id}:"]
            for c in cands:
                types = c.get("types") or []
                if types:
                    types_str = ", ".join(
                        f"{t.get('name') or t.get('id')}{' (' + (t.get('description') or '') + ')' if t.get('description') else ''}"
                        for t in types
                    )
                else:
                    types_str = ""
                lines.append(
                    f"- ID: {c.get('id')} | Name: {c.get('name','')} | Desc: {c.get('description','')}"
                    + (f" | Types: {types_str}" if types_str else "")
                )
            parts.append("\n".join(lines))

        parts.append(
            "Return a JSON object mapping column ids to the chosen Wikidata ID or 'none'.\n"
            'Example: {"0": "Q64", "1": "none"}'
        )
        prompt = "\n\n".join(parts)

        # use same retry/backoff logic as _ask_llm
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "5"))
        backoff = float(os.environ.get("LLM_BACKOFF_INITIAL", "0.5"))

        response_text = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0,
                )
                # extract text similarly
                ans = None
                if hasattr(resp, "choices") and resp.choices:
                    c0 = resp.choices[0]
                    msg = getattr(c0, "message", None)
                    if msg:
                        ans = getattr(msg, "content", None) or (
                            msg.get("content") if isinstance(msg, dict) else None
                        )
                    if not ans:
                        ans = getattr(c0, "text", None)
                if not ans and isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices:
                        c0 = choices[0]
                        if isinstance(c0, dict):
                            ans = (c0.get("message") or {}).get("content") or c0.get("text")

                response_text = str(ans).strip() if ans else None
                if response_text:
                    break
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

        parsed = self._extract_json_from_text(response_text or "")
        if isinstance(parsed, dict):
            # ensure keys are strings
            return {str(k): v for k, v in parsed.items()}

        # fallback: try to parse simple token per column by splitting lines
        results: Dict[str, str] = {}
        if response_text:
            for line in response_text.splitlines():
                # try formats like 0: Q64 or "0": "Q64"
                m = re.match(r"\s*\"?(\d+)\"?\s*[:\-]\s*\"?(Q\d+|none)\"?", line)
                if m:
                    results[m.group(1)] = m.group(2)

        return results


class LLMProcessor(BaseProcessor):
    """
    Processor that uses an LLM (via OpenAI-compatible API) to select the best
    matching candidate for each table cell.

    Configuration via environment variables:
        LLM_BASE_URL  — API base URL (default: https://openrouter.ai/api/v1)
        LLM_API_KEY   — API key
        LLM_MODEL     — Model name (default: openai/gpt-4o-mini)
    """

    processor_id = "llm-processor"

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self.client = OpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ.get("LLM_API_KEY", ""),
        )
        self.model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")

    def process(self, feature):
        """Run the full LLM pipeline in a single call (rank + rerank)."""
        reranker = _LLMReranker(self.config, self.client, self.model)
        reranker.run_llm_process_once()
