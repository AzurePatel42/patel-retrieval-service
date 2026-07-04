from typing import List
from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10


class Chunk(BaseModel):
    chunk_id: int
    text: str
    source: str
    distance: float


class RetrieveResponse(BaseModel):
    query: str
    clean_query: str
    chunks: List[Chunk]
    context: str
