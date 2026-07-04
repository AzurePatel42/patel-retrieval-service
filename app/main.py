from fastapi import FastAPI, HTTPException
from .preprocessing import preprocess_query
from .embedding import embed_query
from .retrieval import retrieve_candidates
from .context import assemble_context
from .db import get_connection
from .rerank import KeywordBoostReranker
from .models import RetrieveRequest, RetrieveResponse


app = FastAPI()

# Choose your reranker (pluggable)
reranker = KeywordBoostReranker()

@app.post("/retrieve")
async def retrieve(query: str, top_k: int = 10):
    # 1. Preprocess
    try:
        clean = preprocess_query(query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Embed
    try:
        vector = await embed_query(clean)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 3. Retrieve from pgvector
    try:
        async with get_connection() as conn:
            candidates = await retrieve_candidates(conn, vector, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # 4. Rerank (new module)
    reranked = reranker.rerank(clean, candidates)

    # 5. Context assembly
    context = assemble_context(reranked)

    # 6. Return final payload
    return {
        "query": query,
        "clean_query": clean,
        "chunks": reranked,
        "context": context,
    }
