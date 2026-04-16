from config import settings
from endpoints.alligator_api import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import RootResponse

_DESCRIPTION = """
## Alligator EMD – REST API

**Alligator** is a semantic annotation engine that automatically links tabular data to
knowledge-graph entities (Wikidata).  This API exposes endpoints to:

* **Upload** tables (JSON or CSV) and trigger background annotation.
* **Browse** datasets and tables with cursor-based pagination.
* **Retrieve** row-level entity annotations (CEA), column-type annotations (CTA) and
  column-property annotations (CPA).
* **Delete** datasets or individual tables.

---

Interactive documentation is available at:
* **Swagger UI** – [`/docs`](/docs)
* **ReDoc** – [`/redoc`](/redoc)
* **OpenAPI JSON** – [`/openapi.json`](/openapi.json)
"""

_TAGS_METADATA = [
    {
        "name": "root",
        "description": "Health-check / service-info endpoint.",
    },
    {
        "name": "datasets",
        "description": "Create, list and delete datasets.",
    },
    {
        "name": "tables",
        "description": "Upload, list, retrieve and delete tables within a dataset.",
    },
]

app = FastAPI(
    title=settings.FASTAPI_APP_NAME,
    description=_DESCRIPTION,
    version="1.0.0",
    debug=settings.DEBUG,
    openapi_tags=_TAGS_METADATA,
    contact={
        "name": "Alligator EMD",
        "url": "https://github.com/unimib-datAI/alligator",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the alligator router
app.include_router(router)


@app.get(
    "/",
    tags=["root"],
    summary="Service info",
    response_model=RootResponse,
    responses={200: {"description": "Basic runtime information about the service."}},
)
def read_root():
    """Return basic runtime configuration of the running service."""
    return {
        "app_name": settings.FASTAPI_APP_NAME,
        "debug": settings.DEBUG,
        "database_url": settings.MONGO_URI,
        "mongo_server_port": settings.MONGO_SERVER_PORT,
        "fastapi_server_port": settings.FASTAPI_SERVER_PORT,
    }
