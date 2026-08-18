import asyncio
from pathlib import Path

from sqlalchemy import text
import sqlparse
from pathlib import Path
import sys
from dotenv import load_dotenv

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from data.db.client import init_async_db

SCHEMA_PATH = Path(__file__).with_name("schema_transformed.sql")


async def run_db():
    db = init_async_db()
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        raw_statements = f.read()

    try:
        async with db.begin() as conn:
            statements = sqlparse.split(raw_statements)
            for statement in statements:
                if statement.strip():
                    await conn.execute(text(statement))
    finally:
        await db.dispose()


if __name__ == "__main__":
    asyncio.run(run_db())
