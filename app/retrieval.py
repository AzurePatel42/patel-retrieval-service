from typing import List, Dict, Any
from asyncpg import Connection


class RetrievalError(Exception):
    pass


RETRIEVAL_SQL = """
SELECT 
    c.id AS chunk_id,
    c.text AS chunk_text,
    d.source_name AS source,
    e.embedding <-> $1 AS distance
FROM embeddings e
JOIN chunks c ON e.chunk_id = c.id
JOIN documents d ON c.document_id = d.id
ORDER BY e.embedding <-> $1
LIMIT $2;
"""


async def retrieve_candidates(
    conn: Connection,
    query_vector: List[float],
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k semantically similar chunks using cosine distance.
    Invariants:
      - Uses <-> operator (cosine distance)
      - Returns deterministic ordering
      - Includes metadata for context assembly
    """

    if not isinstance(query_vector, list) or len(query_vector) == 0:
        raise RetrievalError("Invalid query vector")

    if top_k <= 0:
        raise RetrievalError("top_k must be positive")

    try:
        rows = await conn.fetch(RETRIEVAL_SQL, query_vector, top_k)
    except Exception as e:
        raise RetrievalError(f"Database retrieval failed: {e}") from e

    results = []
    for r in rows:
        results.append({
            "chunk_id": r["chunk_id"],
            "text": r["chunk_text"],
            "source": r["source"],
            "distance": float(r["distance"]),
        })

    return results
