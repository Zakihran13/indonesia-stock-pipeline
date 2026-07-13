from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
import os

def init_async_db(pool_size=50, max_overflow=50) -> AsyncEngine:
    """Returns the db client"""
    
    try:
        user = os.environ["POSTGRESQL_DB_USER"]
        password = os.environ["POSTGRESQL_DB_PASSWORD"]
        host = os.environ["POSTGRESQL_DB_HOST"]
        port = os.environ["POSTGRESQL_DB_PORT"]
        database = os.environ["POSTGRESQL_DB_NAME"]
    except KeyError as e:
        raise ValueError(f"Missing required database environment variable: {e}")

    return create_async_engine(
        f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}",
        pool_size=pool_size,
        max_overflow=max_overflow,
    )

