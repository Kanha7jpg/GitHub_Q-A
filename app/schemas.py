from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=5000)


class RetrievedSnippet(BaseModel):
    source: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    snippets: list[RetrievedSnippet]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ingestion_completed: bool
    error: str | None = None
