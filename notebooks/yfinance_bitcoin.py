import yfinance as yf
import pandas as pd

data = yf.download(
    tickers= "BTC-USD",
    period= "5h",
    interval= "5m"
)

data
pd.DataFrame(data).to_csv("../data/bitcoin_prices.csv")