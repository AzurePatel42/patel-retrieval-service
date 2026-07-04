import os
import math
from typing import List

import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EXPECTED_DIM = 1536


class EmbeddingError(Exception):
    pass


async def embed_query(text: str) -> List[float]:
    """
    Convert a clean query into an embedding vector.
    Invariants:
      - Uses same model as ingestion.
      - Returns a fixed-length float vector.
      - No NaN / None values.
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text")

    if OPENAI_API_KEY is None:
        raise EmbeddingError("OPENAI_API_KEY is not set")

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise EmbeddingError(f"Embedding API request failed: {e}") from e

    data = resp.json()
    try:
        embedding = data["data"][0]["embedding"]
    except (KeyError, IndexError) as e:
        raise EmbeddingError(f"Unexpected embedding response format: {e}") from e

    # Validate type and dimension
    if not isinstance(embedding, list):
        raise EmbeddingError("Embedding is not a list")

    if len(embedding) != EXPECTED_DIM:
        raise EmbeddingError(
            f"Unexpected embedding dimension: {len(embedding)} (expected {EXPECTED_DIM})"
        )

    # Validate numeric values
    for v in embedding:
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise EmbeddingError("Embedding contains invalid numeric values")

    # Cast to float explicitly
    return [float(v) for v in embedding]
