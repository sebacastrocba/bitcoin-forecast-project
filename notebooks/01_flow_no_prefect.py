# GOALS:
# - Recreate the API ETL Script
# - Keep it simple at first (NO forecasting)

import pandas as pd
import yfinance as yf

# 1.0 EXTRACT ----
# - Fetch the bitcoin prices from the yfinance API

def extract_bitcoin_prices() -> pd.DataFrame:
    data = yf.download(
        tickers= "BTC-USD", 
        period= "5h",
        interval= "5m"
    )

    return data

# 2.0 TRANSFORMATION ----
# Not necessary for this example since no transformations,
# we are just saving the results of the extraction

def transform(data: pd.DataFrame) -> pd.DataFrame:
    return data

# 3.0 LOAD ----
# - Store the bitcoin price data in a CSV

def load_bitcoin_prices(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path_or_buf= path, index = True)

# MAIN

if __name__ == '__main__':
    df = extract_bitcoin_prices()
    df = transform(df)
    load_bitcoin_prices(data = df, path= "../data/bitcoin_prices.csv")