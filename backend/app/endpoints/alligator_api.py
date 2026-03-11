import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from bson import ObjectId
from column_classifier import ColumnClassifier  # added global import
from dependencies import get_alligator_db, get_db
from endpoints.imdb_example import IMDB_EXAMPLE  # Example input
from endpoints.response_formatter import build_table_response
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError  # added import
import requests
import time

from alligator import Alligator
from schemas import (
    DatasetCreate,
    DatasetCreateResponse,
    DatasetsListResponse,
    TableAddedResponse,
    TableDataEnvelope,
    TablesListResponse,
)

router = APIRouter()


class TableUpload(BaseModel):
    """JSON body for uploading a table."""

    table_name: str = Field(..., description="Unique table name within the dataset")
    header: List[str] = Field(..., description="List of column names in order")
    total_rows: int = Field(..., description="Total number of data rows")
    classified_columns: Optional[Dict[str, Any]] = Field(
        default={},
        description=(
            "Optional pre-computed column classifications (NE / LIT / IGNORED maps). "
            "If omitted or empty, the ColumnClassifier model is invoked automatically."
        ),
    )
    data: List[dict] = Field(..., description="Table rows as a list of objects")

    model_config = {"json_schema_extra": {"example": IMDB_EXAMPLE}}


# Add helper function to format classification results
def format_classification(raw_classification: dict, header: list) -> dict:
    ne_types = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}
    ne, lit = {}, {}
    for i, col_name in enumerate(header):
        col_result = raw_classification.get(col_name, {})
        classification = col_result.get("classification", "UNKNOWN")
        if classification in ne_types:
            ne[str(i)] = classification
        else:
            lit[str(i)] = classification
    all_indexes = set(str(i) for i in range(len(header)))
    recognized = set(ne.keys()).union(lit.keys())
    ignored = list(all_indexes - recognized)
    return {"NE": ne, "LIT": lit, "IGNORED": ignored}


@router.post(
    "/dataset/{datasetName}/table/json",
    status_code=status.HTTP_201_CREATED,
    response_model=TableAddedResponse,
    tags=["tables"],
    summary="Upload table (JSON)",
    responses={
        201: {"description": "Table created and annotation queued."},
        400: {"description": "Table already exists or duplicate dataset insertion."},
    },
)
def add_table(
    datasetName: str,
    table_upload: TableUpload = Body(...),
    background_tasks: BackgroundTasks = None,
    processor_id: Optional[str] = Query(
        None,
        description="Processor ID to use for annotation (e.g. 'ml-processor', 'llm-processor'). Defaults to the system default.",
    ),
    db: Database = Depends(get_db),
):
    """
    Upload a table as a JSON payload to an existing (or auto-created) dataset and
    trigger Alligator semantic-annotation processing in the background.

    If `classified_columns` is omitted or empty, the built-in
    **ColumnClassifier** model is automatically invoked to infer column types.
    """
    # Check if dataset exists; if not, create it
    dataset = db.datasets.find_one({"dataset_name": datasetName})  # updated query key
    if not dataset:
        try:
            dataset_id = db.datasets.insert_one(
                {
                    "dataset_name": datasetName,  # updated field key
                    "created_at": datetime.now(),
                    "total_tables": 0,
                    "total_rows": 0,
                }
            ).inserted_id
        except DuplicateKeyError:
            raise HTTPException(status_code=400, detail="Duplicate dataset insertion")
    else:
        dataset_id = dataset["_id"]

    if not table_upload.classified_columns:
        df = pd.DataFrame(table_upload.data)
        raw_classification = (
            ColumnClassifier(model_type="fast")
            .classify_multiple_tables([df.head(1024)])[0]
            .get("table_1", {})
        )
        classification = format_classification(raw_classification, table_upload.header)
    else:
        classification = table_upload.classified_columns

    # Create table metadata including classified_columns
    table_metadata = {
        "dataset_name": datasetName,
        "table_name": table_upload.table_name,
        "header": table_upload.header,
        "total_rows": table_upload.total_rows,
        "created_at": datetime.now(),
        "status": "processing",
        "classified_columns": classification,  # added classification field
    }
    try:
        db.tables.insert_one(table_metadata)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=400, detail="Table with this name already exists in the dataset"
        )

    # Update dataset metadata
    db.datasets.update_one(
        {"_id": dataset_id}, {"$inc": {"total_tables": 1, "total_rows": table_upload.total_rows}}
    )

    # Capture names now so the closure doesn't hold the request-scoped db.
    _dataset_name = datasetName
    _table_name = table_upload.table_name
    _processor_id = processor_id

    # Trigger background task with classification passed to Alligator
    def run_alligator_task():
        gator = Alligator(
            input_csv=pd.DataFrame(table_upload.data),
            dataset_name=_dataset_name,
            table_name=_table_name,
            max_candidates=3,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=30,
            model_path="./alligator/models/default.h5",
            save_output_to_csv=False,
            columns_type=classification,
        )
        gator.run(processor_id=_processor_id)
        # Mark the table as fully annotated so polling clients can detect completion.
        # Open a fresh connection — the request-scoped db is already closed at this point.
        from pymongo import MongoClient as _MongoClient

        _client = _MongoClient(os.environ.get("MONGO_URI", "mongodb://gator-mongodb:27017"))
        try:
            _client["alligator_backend_db"].tables.update_one(
                {"dataset_name": _dataset_name, "table_name": _table_name},
                {"$set": {"status": "DONE", "completed_at": datetime.now()}},
            )
        finally:
            _client.close()

    background_tasks.add_task(run_alligator_task)

    return {
        "message": "Table added successfully.",
        "tableName": table_upload.table_name,
        "datasetName": datasetName,
    }


def parse_json_column_classification(column_classification: str = Form("")) -> Optional[dict]:
    # Parse the form field; return None if empty
    if not column_classification:
        return None
    return json.loads(column_classification)


@router.post(
    "/dataset/{datasetName}/table/csv",
    status_code=status.HTTP_201_CREATED,
    response_model=TableAddedResponse,
    tags=["tables"],
    summary="Upload table (CSV)",
    responses={
        201: {"description": "Table created and annotation queued."},
        400: {"description": "Table already exists or duplicate dataset insertion."},
    },
)
def add_table_csv(
    datasetName: str,
    table_name: str = Query(..., description="Unique table name within the dataset"),
    file: UploadFile = File(..., description="CSV file to upload"),
    column_classification: Optional[dict] = Depends(parse_json_column_classification),
    processor_id: Optional[str] = Query(
        None,
        description="Processor ID to use for annotation (e.g. 'ml-processor', 'llm-processor'). Defaults to the system default.",
    ),
    background_tasks: BackgroundTasks = None,
    db: Database = Depends(get_db),
):
    """
    Upload a table as a **CSV file** (multipart/form-data) to an existing (or
    auto-created) dataset and trigger Alligator semantic-annotation in the background.

    Optionally supply `column_classification` as a JSON string form-field to
    override automatic column-type detection.
    """
    # Read CSV file and convert NaN values to None
    df = pd.read_csv(file.file)
    df = df.replace({np.nan: None})  # permanent fix for JSON serialization

    header = df.columns.tolist()
    total_rows = len(df)

    # Use the provided classification; if empty, call ColumnClassifier on a sample
    classification = column_classification if column_classification else {}
    if not classification:
        from column_classifier import ColumnClassifier

        classifier = ColumnClassifier(model_type="fast")
        classification_result = classifier.classify_multiple_tables([df.head(1024)])
        raw_classification = classification_result[0].get("table_1", {})
        classification = format_classification(raw_classification, header)

    # Check if dataset exists; if not, create it
    dataset = db.datasets.find_one({"dataset_name": datasetName})  # updated query key
    if not dataset:
        try:
            dataset_id = db.datasets.insert_one(
                {
                    "dataset_name": datasetName,  # updated field key
                    "created_at": datetime.now(),
                    "total_tables": 0,
                    "total_rows": 0,
                }
            ).inserted_id
        except DuplicateKeyError:
            raise HTTPException(status_code=400, detail="Duplicate dataset insertion")
    else:
        dataset_id = dataset["_id"]

    # Create table metadata
    table_metadata = {
        "dataset_name": datasetName,
        "table_name": table_name,
        "header": header,
        "total_rows": total_rows,
        "created_at": datetime.now(),
        "status": "processing",
        "classified_columns": classification,  # updated field for CSV input
    }
    try:
        db.tables.insert_one(table_metadata)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=400, detail="Table with this name already exists in the dataset"
        )

    # Update dataset metadata
    db.datasets.update_one(
        {"_id": dataset_id}, {"$inc": {"total_tables": 1, "total_rows": total_rows}}
    )

    # Trigger background task with columns_type passed to Alligator
    def run_alligator_task():
        gator = Alligator(
            input_csv=df,
            dataset_name=datasetName,
            table_name=table_name,
            max_candidates=3,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=30,
            model_path="./alligator/models/default.h5",
            save_output_to_csv=False,
            columns_type=classification,
        )
        gator.run(processor_id=processor_id)
        # Mark the table as fully annotated so polling clients can detect completion.
        # Open a fresh connection — the request-scoped db is already closed at this point.
        from pymongo import MongoClient as _MongoClient

        _client = _MongoClient(os.environ.get("MONGO_URI", "mongodb://gator-mongodb:27017"))
        try:
            _client["alligator_backend_db"].tables.update_one(
                {"dataset_name": datasetName, "table_name": table_name},
                {"$set": {"status": "DONE", "completed_at": datetime.now()}},
            )
        finally:
            _client.close()

    background_tasks.add_task(run_alligator_task)

    return {
        "message": "CSV table added successfully.",
        "tableName": table_name,
        "datasetName": datasetName,
    }


@router.get(
    "/datasets",
    response_model=DatasetsListResponse,
    tags=["datasets"],
    summary="List datasets",
    responses={
        200: {"description": "Paginated list of datasets."},
        400: {"description": "Invalid cursor value."},
    },
)
def get_datasets(
    limit: int = Query(10, ge=1, le=200, description="Maximum number of datasets to return"),
    cursor: Optional[str] = Query(
        None, description="Pagination cursor (ObjectId of the last seen document)"
    ),
    db: Database = Depends(get_db),
):
    """
    Return a paginated list of all datasets.

    Results are sorted by internal MongoDB `_id` (ascending).  Pass the
    `next_cursor` value from one response as the `cursor` query parameter in
    the next request to fetch the following page.
    """
    query_filter = {}
    if cursor:
        try:
            query_filter["_id"] = {"$gt": ObjectId(cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    results = db.datasets.find(query_filter).sort("_id", 1).limit(limit)
    datasets = list(results)
    next_cursor = datasets[-1]["_id"] if datasets else None

    for dataset in datasets:
        dataset["_id"] = str(dataset["_id"])
        if "created_at" in dataset:
            dataset["created_at"] = dataset["created_at"].isoformat()

    return {
        "data": datasets,
        "pagination": {
            "next_cursor": (
                str(next_cursor) if next_cursor else None
            )  # removed limit from pagination output
        },
    }


@router.get(
    "/datasets/{dataset_name}/tables",
    response_model=TablesListResponse,
    tags=["tables"],
    summary="List tables in a dataset",
    responses={
        200: {"description": "Paginated list of tables belonging to the dataset."},
        400: {"description": "Invalid cursor value."},
        404: {"description": "Dataset not found."},
    },
)
def get_tables(
    dataset_name: str,
    limit: int = Query(10, ge=1, le=200, description="Maximum number of tables to return"),
    cursor: Optional[str] = Query(
        None, description="Pagination cursor (ObjectId of the last seen document)"
    ),
    db: Database = Depends(get_db),
):
    """
    Return a paginated list of tables that belong to `dataset_name`.

    Each entry includes table-level metadata such as `status`, `total_rows`,
    `header` and `classified_columns`.
    """
    # Ensure dataset exists
    if not db.datasets.find_one({"dataset_name": dataset_name}):  # updated query key
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    query_filter = {"dataset_name": dataset_name}
    if cursor:
        try:
            query_filter["_id"] = {"$gt": ObjectId(cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    results = db.tables.find(query_filter).sort("_id", 1).limit(limit)
    tables = list(results)
    next_cursor = tables[-1]["_id"] if tables else None

    for table in tables:
        table["_id"] = str(table["_id"])
        if "created_at" in table:
            table["created_at"] = table["created_at"].isoformat()
        if "completed_at" in table:
            table["completed_at"] = table["completed_at"].isoformat()

    return {
        "dataset": dataset_name,
        "data": tables,
        "pagination": {
            "next_cursor": (
                str(next_cursor) if next_cursor else None
            )  # removed limit from pagination output
        },
    }


@router.get(
    "/datasets/{dataset_name}/tables/{table_name}",
    response_model=TableDataEnvelope,
    tags=["tables"],
    summary="Get annotated table data",
    responses={
        200: {"description": "Paginated table rows with semantic annotations (CEA / CPA / CTA)."},
        400: {"description": "Invalid cursor value."},
        404: {"description": "Dataset or table not found."},
        500: {"description": "Internal error while formatting the table response."},
    },
)
def get_table(
    dataset_name: str,
    table_name: str,
    limit: int = Query(
        10,
        description="Rows per page. Pass 0 or a negative value to return all rows (no pagination).",
    ),
    cursor: Optional[str] = Query(
        None, description="Pagination cursor (ObjectId of the last seen row)"
    ),
    db: Database = Depends(get_db),
    alligator_db: Database = Depends(get_alligator_db),
):
    """
    Return annotated rows for a single table together with full semantic
    annotations:

    * **CEA** – Cell-Entity Annotations (entity candidates per cell).
    * **CPA** – Column-Property Annotations (predicates between column pairs).
    * **CTA** – Column-Type Annotations (Wikidata types per NE column).

    Use `limit` and `cursor` for cursor-based pagination.
    """
    # Check dataset
    if not db.datasets.find_one({"dataset_name": dataset_name}):  # updated query key
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Check table
    table = db.tables.find_one({"dataset_name": dataset_name, "table_name": table_name})
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    header = table.get("header", [])

    # Pagination filter
    query_filter = {"dataset_name": dataset_name, "table_name": table_name}
    if cursor:
        try:
            query_filter["_id"] = {"$gt": ObjectId(cursor)}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor value")

    # Fetch rows from the Alligator-processed data
    finder = alligator_db.input_data.find(query_filter).sort("_id", 1)
    # if limit <= 0 return all rows (no pagination)
    if limit and int(limit) > 0:
        results = finder.limit(limit)
    else:
        results = finder
    raw_rows = list(results)
    # Delegate formatting and aggregation to the response formatter
    try:
        response = build_table_response(raw_rows, table, dataset_name, header)
    except Exception as e:
        print(f"Error building table response: {e}")
        raise HTTPException(status_code=500, detail="Failed to format table response")
    # keep previous API shape for compatibility: wrap under `data`
    return {"data": response}


@router.post(
    "/datasets",
    status_code=status.HTTP_201_CREATED,
    response_model=DatasetCreateResponse,
    tags=["datasets"],
    summary="Create dataset",
    responses={
        201: {"description": "Dataset created successfully."},
        400: {"description": "A dataset with the given name already exists."},
    },
)
def create_dataset(
    dataset_data: DatasetCreate = Body(...),
    db: Database = Depends(get_db),
):
    """
    Create a new, empty dataset.

    The `dataset_name` must be unique across all datasets.  Once created, tables
    can be uploaded via the `/dataset/{datasetName}/table/json` or
    `/dataset/{datasetName}/table/csv` endpoints.
    """
    dataset_dict = dataset_data.model_dump()
    existing = db.datasets.find_one(
        {"dataset_name": dataset_dict.get("dataset_name")}
    )  # updated query key
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset with dataset_name {dataset_dict.get('dataset_name')} already exists",
        )

    dataset_dict["created_at"] = datetime.now()
    dataset_dict["total_tables"] = 0
    dataset_dict["total_rows"] = 0

    try:
        result = db.datasets.insert_one(dataset_dict)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Dataset already exists")
    dataset_dict["_id"] = str(result.inserted_id)

    return {"message": "Dataset created successfully", "dataset": dataset_dict}


@router.delete(
    "/datasets/{dataset_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["datasets"],
    summary="Delete dataset",
    responses={
        204: {"description": "Dataset and all its tables deleted."},
        404: {"description": "Dataset not found."},
    },
)
def delete_dataset(
    dataset_name: str,
    db: Database = Depends(get_db),
    alligator_db: Database = Depends(get_alligator_db),
):
    """
    Permanently delete a dataset and **all tables** that belong to it.

    This operation is irreversible.
    """
    # Check existence using uniform dataset key
    existing = db.datasets.find_one({"dataset_name": dataset_name})  # updated query key
    if not existing:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    # Delete all tables associated with this dataset
    db.tables.delete_many({"dataset_name": dataset_name})

    # Delete dataset
    db.datasets.delete_one({"dataset_name": dataset_name})  # updated query key

    # Optionally delete data from alligator_db if needed
    return None


@router.delete(
    "/datasets/{dataset_name}/tables/{table_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["tables"],
    summary="Delete table",
    responses={
        204: {"description": "Table deleted."},
        404: {"description": "Dataset or table not found."},
    },
)
def delete_table(dataset_name: str, table_name: str, db: Database = Depends(get_db)):
    """
    Permanently delete a single table from a dataset.

    The parent dataset's `total_tables` and `total_rows` counters are updated
    automatically.  This operation is irreversible.
    """
    # Ensure dataset exists using uniform dataset key
    dataset = db.datasets.find_one({"dataset_name": dataset_name})  # updated query key
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    table = db.tables.find_one({"dataset_name": dataset_name, "table_name": table_name})
    if not table:
        raise HTTPException(
            status_code=404, detail=f"Table {table_name} not found in dataset {dataset_name}"
        )

    row_count = table.get("total_rows", 0)

    # Delete table
    db.tables.delete_one({"dataset_name": dataset_name, "table_name": table_name})

    # Update dataset metadata
    db.datasets.update_one(
        {"name": dataset_name}, {"$inc": {"total_tables": -1, "total_rows": -row_count}}
    )

    # Optionally delete data from alligator_db if needed
    return None
