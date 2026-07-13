import pandas as pd
from sqlalchemy.util import symbol
import yfinance as yf

from functools import partial
import asyncio
import aiometer
import nest_asyncio
nest_asyncio.apply()


from data.db.statements import insert_metadata
from utils.helper import split_batch
from data.db.client import init_async_db



def process_ticker(ticker: list[str]):
    """fetches the metadata for a given ticker using yfinance and returns a DataFrame."""

    flatten_ticker = " ".join(ticker)
    metadata_list = []

    try:
        tickers = yf.Tickers(flatten_ticker)
        for s, t in tickers.tickers.items():
            metadata_list.append({
                "Symbol": s,
                **t.info
            })

    except Exception as e:
        print(f"Error fetching metadata for {flatten_ticker}: {e}")

    return pd.DataFrame(metadata_list)




async def exec_metadata():
    all_tickers = pd.read_json(r"C:\Users\zakis\OneDrive\Desktop\indonesia-stock-pipeline\data\indonesian_stock_list.json")
    all_tickers['Kode Jakarta'] = all_tickers['Kode'] + ".JK"

    batches = split_batch(all_tickers["Kode Jakarta"].tolist()[:20], 50)

    tasks = [asyncio.to_thread(process_ticker, batch) for batch in batches]
    results = await asyncio.gather(*tasks)
    metadata_df = pd.concat(results, ignore_index=True)

    # store data
    engine = init_async_db()
    try:
        async with engine.begin() as conn:
            for _, row in metadata_df.iterrows():
                await insert_metadata(conn, row.to_dict()) 
    finally:
        await engine.dispose()



if __name__ == "__main__":
    asyncio.run(exec_metadata())