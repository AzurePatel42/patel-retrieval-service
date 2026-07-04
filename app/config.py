import os
from pydantic import BaseSettings, AnyUrl


class Settings(BaseSettings):
    # Postgres with pgvector
    DATABASE_URL: AnyUrl = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"

    # OpenAI
    OPENAI_API_KEY: str | None = None

    # Retrieval defaults
    TOP_K_DEFAULT: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
