import pandas as pd
import yfinance as yf
from prefect import task, flow 

# 1.0 EXTRACT ----
# - Fetch the bitcoin prices from the yfinance API

@task
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

@task
def transform(data: pd.DataFrame) -> pd.DataFrame:
    return data

# 3.0 LOAD ----
# - Store the bitcoin price data in a CSV

@task
def load_bitcoin_prices(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path_or_buf= path, index = True)

# 4.0 PREFECT FLOW ----
@flow
def main_flow(log_prints = True):
    print(">>> Extracting Bitcoin Prices")
    df = extract_bitcoin_prices()
    print(">>> Doing the Transform")
    df = transform(df)
    print(">>> Storing bitcoin prices")
    load_bitcoin_prices(
        data=df,
        path = "../data/bitcoin_prices.csv"
    )

# 5.0 MAIN ----
if __name__ == '__main__':
    main_flow()