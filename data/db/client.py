from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
import os
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ConfigurationError


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


class AsyncMongoDBConnection:
    """A singleton-like manager for the Motor async connection pool."""

    _client: AsyncIOMotorClient | None = None

    @classmethod
    def connect(cls) -> None:
        """Initializes the async client. Motor does this without blocking."""
        if cls._client is not None:
            return

        uri = os.getenv("MONGO_URI")
        if not uri:
            raise ConfigurationError("MONGO_URI environment variable is not set.")

        # Motor initializes instantly; it connects to nodes in the background
        cls._client = AsyncIOMotorClient(
            uri, maxPoolSize=50, serverSelectionTimeoutMS=5000
        )
        logger.info("Async MongoDB connection pool initialized.")

    @classmethod
    async def verify_connection(cls) -> None:
        """Explicitly pings the server asynchronously to fail-fast if unreachable."""
        client = cls.get_client()
        try:
            # We await the ping command to verify the network
            await client.admin.command("ping")
            logger.info("Successfully established connection to MongoDB.")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        if cls._client is None:
            cls.connect()

        if cls._client is None:
            raise ConnectionFailure("MongoDB client is not initialized.")

        return cls._client

    @classmethod
    def close(cls):
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            logger.info("Async MongoDB connection closed.")


def get_async_mongodb(db_name: str):
    client = AsyncMongoDBConnection.get_client()
    return client[db_name]
