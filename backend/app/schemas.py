"""
Pydantic schemas used across the Alligator API.

These models drive FastAPI's automatic OpenAPI / Swagger documentation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generic / shared
# ---------------------------------------------------------------------------


class PaginationInfo(BaseModel):
    next_cursor: Optional[str] = Field(None, description="Cursor for the next page of results")


class MessageResponse(BaseModel):
    message: str = Field(..., description="Human-readable status message")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class RootResponse(BaseModel):
    app_name: str = Field(..., description="Application name")
    debug: bool = Field(..., description="Whether the app is running in debug mode")
    database_url: str = Field(..., description="MongoDB connection URI")
    mongo_server_port: int = Field(..., description="MongoDB server port")
    fastapi_server_port: int = Field(..., description="FastAPI server port")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DatasetInfo(BaseModel):
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")
    dataset_name: str = Field(..., description="Unique dataset identifier")
    created_at: Optional[str] = Field(None, description="ISO-formatted creation timestamp")
    total_tables: int = Field(0, description="Number of tables in the dataset")
    total_rows: int = Field(0, description="Total rows across all tables in the dataset")

    class Config:
        populate_by_name = True


class DatasetCreate(BaseModel):
    dataset_name: str = Field(..., description="Unique name for the new dataset")

    model_config = {
        "json_schema_extra": {
            "example": {"dataset_name": "my_dataset"}
        }
    }


class DatasetCreateResponse(BaseModel):
    message: str = Field(..., description="Human-readable status message")
    dataset: Dict[str, Any] = Field(..., description="Created dataset document")


class DatasetsListResponse(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of dataset documents")
    pagination: PaginationInfo


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class ClassifiedColumns(BaseModel):
    NE: Dict[str, str] = Field(
        default_factory=dict,
        description="Named-Entity columns: column-index → NE type",
    )
    LIT: Dict[str, str] = Field(
        default_factory=dict,
        description="Literal columns: column-index → literal type",
    )
    IGNORED: List[str] = Field(
        default_factory=list,
        description="Column indexes to ignore during annotation",
    )


class TableInfo(BaseModel):
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")
    dataset_name: str = Field(..., description="Parent dataset name")
    table_name: str = Field(..., description="Unique table name within the dataset")
    header: List[str] = Field(..., description="Column names in order")
    total_rows: int = Field(..., description="Total number of rows in the table")
    created_at: Optional[str] = Field(None, description="ISO-formatted creation timestamp")
    completed_at: Optional[str] = Field(None, description="ISO-formatted annotation completion timestamp")
    status: str = Field(..., description="Processing status: 'processing' | 'DONE'")
    classified_columns: Optional[ClassifiedColumns] = Field(
        None, description="Column type classification used by Alligator"
    )

    class Config:
        populate_by_name = True


class TablesListResponse(BaseModel):
    dataset: str = Field(..., description="Dataset name")
    data: List[Dict[str, Any]] = Field(..., description="List of table documents")
    pagination: PaginationInfo


class TableAddedResponse(BaseModel):
    message: str = Field(..., description="Human-readable status message")
    tableName: str = Field(..., description="Name of the newly added table")
    datasetName: str = Field(..., description="Name of the parent dataset")


# ---------------------------------------------------------------------------
# Table upload (JSON body)
# ---------------------------------------------------------------------------


class TableUploadBody(BaseModel):
    """Request body for uploading a table as JSON."""

    table_name: str = Field(..., description="Unique table name within the dataset")
    header: List[str] = Field(..., description="List of column names")
    total_rows: int = Field(..., description="Total number of data rows")
    classified_columns: Optional[ClassifiedColumns] = Field(
        None,
        description=(
            "Optional pre-computed column classifications. "
            "If omitted, the ColumnClassifier model is called automatically."
        ),
    )
    data: List[Dict[str, Any]] = Field(..., description="Table rows as a list of dicts")

    model_config = {
        "json_schema_extra": {
            "example": {
                "table_name": "movies",
                "header": ["title", "year", "director"],
                "total_rows": 2,
                "classified_columns": {},
                "data": [
                    {"title": "Inception", "year": "2010", "director": "Christopher Nolan"},
                    {"title": "The Matrix", "year": "1999", "director": "Wachowski Sisters"},
                ],
            }
        }
    }


# ---------------------------------------------------------------------------
# Table data (GET /datasets/{dataset}/tables/{table})
# ---------------------------------------------------------------------------


class EntityAnnotation(BaseModel):
    id: str = Field(..., description="Wikidata entity QID")
    name: Optional[str] = Field(None, description="Entity label")
    score: Optional[float] = Field(None, description="Confidence score (0–1)")
    match: bool = Field(False, description="Whether this entity is considered the best match")


class CEAAnnotation(BaseModel):
    idColumn: Any = Field(..., description="Column index")
    idRow: Any = Field(..., description="Row identifier")
    entities: List[EntityAnnotation] = Field(default_factory=list)


class CPAAnnotation(BaseModel):
    idSourceColumn: Any = Field(..., description="Source column index")
    idTargetColumn: Any = Field(..., description="Target column index")
    predicates: List[Dict[str, Any]] = Field(default_factory=list)


class CTAAnnotation(BaseModel):
    idColumn: Any = Field(..., description="Column index")
    types: List[Dict[str, Any]] = Field(default_factory=list)


class SemanticAnnotations(BaseModel):
    cea: List[CEAAnnotation] = Field(default_factory=list, description="Cell-Entity Annotations")
    cpa: List[CPAAnnotation] = Field(default_factory=list, description="Column-Property Annotations")
    cta: List[CTAAnnotation] = Field(default_factory=list, description="Column-Type Annotations")


class RowData(BaseModel):
    idRow: Any = Field(..., description="Row identifier")
    data: List[Any] = Field(default_factory=list, description="Raw cell values")
    linked_entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Entities linked to NE columns in this row"
    )
    status: Optional[str] = Field(None, description="Per-row annotation status")


class ColumnTag(BaseModel):
    idColumn: Any = Field(..., description="Column index")
    tag: str = Field(..., description="Column role tag: SUBJ | NE | LIT")


class TableMetadata(BaseModel):
    column: List[ColumnTag] = Field(
        default_factory=list,
        description="Per-column tag annotations (SUBJ / NE / LIT).",
    )


class TableDataResponse(BaseModel):
    datasetName: str = Field(..., description="Parent dataset name")
    tableName: str = Field(..., description="Table name")
    header: List[str] = Field(..., description="Column names in order")
    rows: List[RowData] = Field(default_factory=list)
    semanticAnnotations: SemanticAnnotations
    metadata: Optional[TableMetadata] = None
    status: Optional[str] = None
    pagination: Optional[PaginationInfo] = None


class TableDataEnvelope(BaseModel):
    """Outer envelope wrapping TableDataResponse under the 'data' key."""

    data: TableDataResponse
