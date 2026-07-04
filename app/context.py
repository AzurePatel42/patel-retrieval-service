import tiktoken
from typing import List, Dict

ENCODER = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 1500  # adjustable depending on your LLM


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def assemble_context(chunks: List[Dict], max_tokens: int = MAX_TOKENS) -> str:
    """
    Build the final context window for the LLM.
    Invariants:
      - Token-limited
      - Ordered by document position
      - Deduplicated
      - Cleanly formatted
    """

    # 1. Sort by chunk_id (document order)
    chunks = sorted(chunks, key=lambda c: c["chunk_id"])

    # 2. Deduplicate identical text
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c["text"] not in seen:
            unique_chunks.append(c)
            seen.add(c["text"])

    # 3. Build context with token budgeting
    final_parts = []
    total_tokens = 0

    for c in unique_chunks:
        chunk_text = c["text"].strip()
        chunk_tokens = count_tokens(chunk_text)

        # Stop if adding this chunk exceeds the budget
        if total_tokens + chunk_tokens > max_tokens:
            break

        # Add metadata header
        header = f"### Source: {c['source']} (chunk {c['chunk_id']})"
        header_tokens = count_tokens(header)

        if total_tokens + header_tokens + chunk_tokens > max_tokens:
            break

        final_parts.append(header)
        final_parts.append(chunk_text)

        total_tokens += header_tokens + chunk_tokens

    return "\n\n".join(final_parts)
