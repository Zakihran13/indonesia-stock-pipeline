import asyncio
from pathlib import Path

from sqlalchemy import text
import sqlparse

from client import init_async_db
from dotenv import load_dotenv


load_dotenv(".env")


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


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