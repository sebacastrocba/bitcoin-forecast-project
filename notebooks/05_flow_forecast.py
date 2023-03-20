# GOALS
# - Put everything together and ...
# - make a forecast using modeltime engine

import pandas as pd
import yfinance as yf 
from prefect import flow, task

from modeltime_forecast import make_modeltime_forecast
from multiprocessing import Process

# 1.0 EXTRACT ----
# - Fetch the bitcoin prices from the yfinance API
# - If fails, retry the API twice (3-second delay) 
@task(
    name="Extract Bitcoin Prices",
    retries=2,
    retry_delay_seconds=3
)
def extract_bitcoin_prices(tickers: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        tickers  = tickers,
        period   = period,
        interval = interval
    )
    return data

# 2.0 LOAD ----
# - Store the bitcoin price data in a CSV
@task(name="Save Bitcoin Price Data as CSV")
def load_bitcoin_prices(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path_or_buf=path, index=True)

# - Make the Forecasts with modeltime_forecast.py
@task(name="Forecast Bitcoin Prices")
def load_bitcoin_forecasts(path_1: str, path_2: str) -> None:
    r_process = Process(
        target=make_modeltime_forecast, 
        kwargs = dict(path_1=path_1, path_2=path_2)
    )
    r_process.start()

# 4.0 PREFECT FLOW -----
@flow(name="bitcoin_forecast_pipeline")
def main_forecast_flow(
    tickers  = "BTC-USD",
    period   = "5h",
    interval = "5m",
    path_bitcoin_prices = '../data/bitcoin_prices.csv',
    path_bitcoin_forecasts = '../data/bitcoin_prices_forecast.csv'
):
    print(">>> Extracting Bitcoin Prices")
    df = extract_bitcoin_prices(
        tickers=tickers, period=period, interval=interval
    )
    print(f">>> Storing Bitcoin Prices: {path_bitcoin_prices}")
    load_bitcoin_prices(
        data=df, 
        path=path_bitcoin_prices
    ) 
    print(f">>> Making Forecast")
    load_bitcoin_forecasts(
        path_1=path_bitcoin_prices,
        path_2=path_bitcoin_forecasts
    )
    print(f">>> Forecast Stored: {path_bitcoin_forecasts}")

# 5.0 MAIN PROGRAM ----
if __name__ == "__main__":
    main_forecast_flow(
        tickers  = "BTC-USD",
        period   = "5h",
        interval = "5m",
        # WARNING: Relative paths won't work with deployments
        #  Solution is to override the parameters in the 
        #  deployment.YAML file with the absolute path
        path_bitcoin_prices  = '../data/bitcoin_prices.csv',
        path_bitcoin_forecasts = '../data/bitcoin_prices_forecast.csv'
    )
    
# TESTING: 
#  python 03_Prefect/flow_05_forecast_scheduler.py

# DEPLOYMENT STEPS & CLI COMMANDS:
#  1. BUILD: 
#      prefect deployment build ./03_Prefect/flow_05_forecast_scheduler.py:main_forecast_flow --name bitcoin_forecast_flow --interval 60
#  2. parameters: 
#       path_bitcoin_prices: '/home/seba/Documentos/Proyectos_personales/bitcoin-project/data/bitcoin_prices.csv'
#       path_bitcoin_forecasts: "/home/seba/Documentos/Proyectos_personales/bitcoin-project/data/bitcoin_prices_forecast.csv"
#  3. APPLY: 
#      prefect deployment apply main_forecast_flow-deployment.yaml
#  4. LIST DEPLOYMENTS: 
#      prefect deployment ls
#  5. RUN: 
#      prefect deployment run "bitcoin_forecast_pipeline/bitcoin_forecast_flow" 
#  6. ORION GUI: 
#      prefect orion start
#  7. AGENT START: 
#      prefect agent start  --work-queue "default"
#  8. Ctrl + C to exit


