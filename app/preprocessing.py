import re
import unicodedata

def preprocess_query(raw_query: str) -> str:
    """
    Clean and normalize the user query before embedding.
    Invariant: Only semantic meaning goes forward.
    """

    if raw_query is None:
        raise ValueError("Query cannot be empty")

    # 1. Normalize unicode (removes weird characters)
    q = unicodedata.normalize("NFKC", raw_query)

    # 2. Strip leading/trailing whitespace
    q = q.strip()

    # 3. Collapse multiple spaces/newlines/tabs
    q = re.sub(r"\s+", " ", q)

    # 4. Remove emojis & non‑semantic symbols
    q = re.sub(r"[^\w\s.,?]", "", q)

    # 5. Lowercase normalization
    q = q.lower()

    # 6. Enforce minimum semantic length
    if len(q) < 3:
        raise ValueError("Query too short to embed meaningfully")

    return q
