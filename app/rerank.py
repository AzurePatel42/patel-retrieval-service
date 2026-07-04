from typing import List, Dict
import math


class RerankError(Exception):
    pass


class BaseReranker:
    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        raise NotImplementedError
        

class DistanceReranker(BaseReranker):
    """
    Default reranker: keep pgvector order (cosine distance).
    This is effectively a no-op but keeps the interface clean.
    """
    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        # Assume candidates already sorted by distance ascending
        return candidates


class KeywordBoostReranker(BaseReranker):
    """
    Simple heuristic reranker:
    - Base score from cosine distance (lower is better)
    - Boost if query keywords appear in the chunk text
    """

    def __init__(self, distance_weight: float = 1.0, keyword_weight: float = 0.5):
        self.distance_weight = distance_weight
        self.keyword_weight = keyword_weight

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not isinstance(candidates, list):
            raise RerankError("Candidates must be a list")

        query_terms = {t for t in query.lower().split() if len(t) > 2}

        scored = []
        for c in candidates:
            text = c.get("text", "") or ""
            distance = float(c.get("distance", math.inf))

            # Base score: inverse of distance (smaller distance → higher score)
            base_score = -distance * self.distance_weight

            # Keyword overlap boost
            text_terms = set(text.lower().split())
            overlap = query_terms.intersection(text_terms)
            keyword_score = len(overlap) * self.keyword_weight

            total_score = base_score + keyword_score

            scored.append((total_score, c))

        # Sort by score descending (higher score = more relevant)
        scored.sort(key=lambda x: x[0], reverse=True)

        return [c for _, c in scored]
