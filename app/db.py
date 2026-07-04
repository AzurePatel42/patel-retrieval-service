import asyncpg
from contextlib import asynccontextmanager
from .config import settings


async def create_pool():
    return await asyncpg.create_pool(str(settings.DATABASE_URL))


_pool = None


async def get_pool():
    global _pool
    if _pool is None:
        _pool = await create_pool()
    return _pool


@asynccontextmanager
async def get_connection():
    pool = await get_pool()
    conn = await pool.acquire()
    try:
        yield conn
    finally:
        await pool.release(conn)
