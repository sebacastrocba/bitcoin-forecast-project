# GOALS:
# - Make a deployment YAML file
# - Expose Scheduling to run the flow on an "interval schedule"
# - Can also do cron schedule

import pandas as pd
import yfinance as yf 
from prefect import task, flow 

# 1.0 EXTRACT ----
# - Fetch the bitcoin prices from the yfinance API

@task(
        name = "Extract Bitcoin Prices",
        retries= 2,
        retry_delay_seconds= 3
)
def extract_bitcoin_prices(tickers: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        tickers= tickers, 
        period= period,
        interval= interval
    )

    return data

# 2.0 LOAD ----
# - Store the bitcoin price data in a CSV

@task(name = "Save Bitcoin Price Data as CSV")
def load_bitcoin_prices(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path_or_buf= path, index = True)

# 4.0 PREFECT FLOW ----
@flow(name = "Bitcoin Price Pipeline")
def main_flow(
    tickers = "BTC-USD",
    period = "5h",
    interval = "5m",
    path = "../data/bitcoin_prices.csv"
):
    print(">>> Extracting Bitcoin Prices")
    df = extract_bitcoin_prices(tickers = tickers, period=period, interval=interval)
    
    print(">>> Storing bitcoin prices")
    load_bitcoin_prices(
        data=df,
        path = path
    )

# 5.0 MAIN PROGRAM ----
if __name__ == '__main__':
    main_flow(
        tickers="BTC-USD",
        period="5h",
        interval="5m",
        path= "../data/bitcoin_prices.csv"
    )