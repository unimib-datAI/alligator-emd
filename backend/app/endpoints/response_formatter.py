from typing import Any, Dict, List
import os
import requests
import time


def build_table_response(raw_rows: List[Dict[str, Any]], table: Dict[str, Any], dataset_name: str, header: List[str]):
    """Build API response matching sintef example format.
    Returns a dict with keys: datasetName, tableName, header, rows, semanticAnnotations, metadata, status, pagination
    """
    # rows: idRow (int when possible) + data
    rows_formatted = []
    for row in raw_rows:
        rid = row.get("row_id")
        try:
            rid_val = int(rid) if isinstance(rid, str) and rid.isdigit() else rid
        except Exception:
            rid_val = rid

        # If the row doesn't include a `linked_entities` field, synthesize it from `cea`
        linked = row.get("linked_entities")
        if not linked:
            linked = []
            cea = row.get("cea", {}) or {}
            for col_str, entities in cea.items():
                if not entities:
                    continue
                try:
                    col_idx = int(col_str)
                except Exception:
                    col_idx = col_str
                # extract entity ids (handles list of dicts or strings)
                ent_ids = []
                for e in entities:
                    if isinstance(e, dict):
                        eid = e.get("id") or e.get("qid") or e.get("qid")
                    else:
                        eid = e
                    if eid:
                        ent_ids.append(eid)
                if ent_ids:
                    linked.append({"idColumn": col_idx, "entities": ent_ids})

        # include row-level status so clients can detect per-row completion
        rows_formatted.append({
            "idRow": rid_val,
            "data": row.get("data", []),
            "linked_entities": linked,
            "status": row.get("status")
        })

    # build semantic annotations
    semantic_annotations = {"cea": [], "cpa": [], "cta": []}
    _predicate_cache: Dict[str, Dict[str, Any]] = {}

    # cea
    for row in raw_rows:
        row_id = row.get("row_id")
        try:
            rdf_row_id = int(row_id) if isinstance(row_id, str) and row_id.isdigit() else row_id
        except Exception:
            rdf_row_id = row_id
        cea = row.get("cea", {}) or {}
        for col_str, entities in cea.items():
            try:
                col_idx = int(col_str)
            except Exception:
                col_idx = col_str

            # ensure each entity has an explicit boolean `match` field
            # use environment variable MATCH_THRESHOLD (default 0.5) to decide match by score
            try:
                MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.5"))
            except Exception:
                MATCH_THRESHOLD = 0.5
            normalized_entities = []
            for ent in (entities or []):
                if isinstance(ent, dict):
                    # preserve existing explicit match value
                    if ent.get("match") is None:
                        score = ent.get("score")
                        try:
                            s = float(score) if score is not None else None
                        except Exception:
                            s = None
                        ent["match"] = True if (s is not None and s >= MATCH_THRESHOLD) else False
                    normalized_entities.append(ent)
                else:
                    # non-dict entity: wrap into a dict without match (cannot infer score)
                    normalized_entities.append({"id": ent, "match": False})

            semantic_annotations["cea"].append(
                {"idColumn": col_idx, "idRow": rdf_row_id, "entities": normalized_entities}
            )

    # cpa aggregation
    cpa_map = {}
    for row in raw_rows:
        cpa = row.get("cpa", {}) or {}
        for src, targets in cpa.items():
            for tgt, preds in targets.items():
                try:
                    s = int(src)
                except Exception:
                    s = src
                try:
                    t = int(tgt)
                except Exception:
                    t = tgt
                key = (s, t)
                if key not in cpa_map:
                    cpa_map[key] = {}

                if isinstance(preds, dict):
                    for pid, val in preds.items():
                        if isinstance(val, dict):
                            score = val.get("score")
                            name = val.get("name")
                        else:
                            try:
                                score = float(val)
                            except Exception:
                                score = None
                            name = None
                        existing = cpa_map[key].get(pid)
                        if (
                            existing is None
                            or (score is not None and (existing.get("score") is None or score > existing.get("score")))
                        ):
                            cpa_map[key][pid] = {"id": pid, "name": name, "score": score}
                elif isinstance(preds, list):
                    for p in preds:
                        if isinstance(p, dict):
                            pid = p.get("id") or p.get("predicate")
                            score = p.get("score")
                            name = p.get("name")
                        else:
                            pid = p
                            score = None
                            name = None
                        existing = cpa_map[key].get(pid)
                        if (
                            existing is None
                            or (score is not None and (existing.get("score") is None or score > existing.get("score")))
                        ):
                            cpa_map[key][pid] = {"id": pid, "name": name, "score": score}
                else:
                    pid = preds
                    existing = cpa_map[key].get(pid)
                    if existing is None:
                        cpa_map[key][pid] = {"id": pid, "name": None, "score": None}

    # batch fetch names
    def _batch_fetch_predicate_names(pids: List[str]):
        pids = [p for p in pids if isinstance(p, str) and p]
        if not pids:
            return
        chunk_size = 50
        headers = {"User-Agent": "semTUI/1.0 (+https://example.org)", "Accept": "application/json"}
        for i in range(0, len(pids), chunk_size):
            chunk = pids[i : i + chunk_size]
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    resp = requests.get(
                        "https://www.wikidata.org/w/api.php",
                        params={"action": "wbgetentities", "ids": "|".join(chunk), "props": "labels", "languages": "en", "format": "json"},
                        headers=headers,
                        timeout=5,
                    )
                    if resp.status_code != 200:
                        raise Exception(f"Wikidata API {resp.status_code}")
                    data = resp.json()
                    entities = data.get("entities", {})
                    for pid in chunk:
                        ent = entities.get(pid, {})
                        labels = ent.get("labels", {})
                        name = None
                        if labels and "en" in labels:
                            name = labels["en"].get("value")
                        existing = _predicate_cache.get(pid, {})
                        score = existing.get("score") if isinstance(existing, dict) else None
                        _predicate_cache[pid] = {"id": pid, "name": name, "score": score}
                    break
                except Exception:
                    if attempt < max_retries:
                        time.sleep(0.5 * attempt)
                    else:
                        for pid in chunk:
                            try:
                                alt = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{pid}.json", headers=headers, timeout=5)
                                if alt.status_code == 200:
                                    j = alt.json()
                                    ent = j.get("entities", {}).get(pid, {})
                                    labels = ent.get("labels", {})
                                    name = None
                                    if labels and "en" in labels:
                                        name = labels["en"].get("value")
                                else:
                                    name = None
                            except Exception:
                                name = None
                            existing = _predicate_cache.get(pid, {})
                            score = existing.get("score") if isinstance(existing, dict) else None
                            _predicate_cache[pid] = {"id": pid, "name": name, "score": score}

    missing_pids = []
    for preds_dict in cpa_map.values():
        for pid, pred in preds_dict.items():
            if pred.get("name") is None:
                missing_pids.append(pid)
    if missing_pids:
        _batch_fetch_predicate_names(list(dict.fromkeys(missing_pids)))

    for (s, t), preds_dict in cpa_map.items():
        pred_list = list(preds_dict.values())
        for pred in pred_list:
            if pred.get("name") is None and pred.get("id"):
                cached = _predicate_cache.get(pred["id"]) or {}
                pred["name"] = cached.get("name")
        semantic_annotations["cpa"].append({"idSourceColumn": s, "idTargetColumn": t, "predicates": pred_list})

    # cta
    cta_map = {}
    for row in raw_rows:
        cta = row.get("cta", {}) or {}
        for col, qids in cta.items():
            try:
                c = int(col)
            except Exception:
                c = col
            cta_map.setdefault(c, set()).update(qids or [])

    for c, qids in cta_map.items():
        semantic_annotations["cta"].append({"idColumn": c, "types": [{"id": q} for q in qids]})

    next_cursor = str(raw_rows[-1]["_id"]) if raw_rows else None

    # metadata
    metadata_columns = []
    classified = table.get("classified_columns", {}) or {}
    ne_cols = set(classified.get("NE", {}).keys())
    lit_cols = set(classified.get("LIT", {}).keys())
    for i in range(len(header)):
        if i == 0:
            tag = "SUBJ"
        elif str(i) in ne_cols:
            tag = "NE"
        elif str(i) in lit_cols:
            tag = "LIT"
        else:
            tag = "LIT"
        metadata_columns.append({"idColumn": i, "tag": tag})

    return {
        "datasetName": dataset_name,
        "tableName": table.get("table_name"),
        "header": header,
        "rows": rows_formatted,
        "semanticAnnotations": semantic_annotations,
        "metadata": {"column": metadata_columns},
        "status": table.get("status", "DONE"),
        "pagination": {"next_cursor": next_cursor},
    }
