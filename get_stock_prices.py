import argparse
from datetime import datetime
from os import path

import requests
import pandas as pd
import io

def preprocess_stock_prices(stock_prices):
    stock_price_changes = []
    for i in range(stock_prices.shape[0] - 1):
        stock_price_changes.append((stock_prices.loc[i, "adjusted_close"] - stock_prices.loc[i + 1, "adjusted_close"]) / stock_prices.loc[i + 1, "adjusted_close"])
    stock_price_changes.append(0.0)
    stock_prices["change"] = stock_price_changes

    return stock_prices

def get_stock_prices(stock_code):
    with open("./credentials/alpha_vantage_api_key.txt", "r") as f:
        api_key = f.read()

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock_code,
        "outputsize": "full",
        "datatype": "csv",
        "apikey": api_key
    }
    api_result = requests.get("https://www.alphavantage.co/query", params=params)

    stock_prices = pd.read_csv(io.StringIO(api_result.text))

    stock_prices = preprocess_stock_prices(stock_prices)

    if path.isfile("./data/stock_prices/" + stock_code + ".csv"):
        old_stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv")
        latest_date = old_stock_prices.loc[0, "timestamp"]
        stock_prices = stock_prices[stock_prices["timestamp"] > latest_date]
        stock_prices = stock_prices.append(old_stock_prices, ignore_index=True)

    stock_prices.to_csv("./data/stock_prices/" + stock_code + ".csv", index=False)
    
    weekdays = stock_prices["timestamp"].apply(lambda timestamp: datetime.strptime(timestamp, "%Y-%m-%d").weekday())
    weekly_stock_prices = stock_prices[weekdays == 4]
    weekly_stock_prices.to_csv("./data/stock_prices/" + stock_code + "_weekly.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get stock prices for stocks.")
    parser.add_argument("stock_codes", nargs="+", help="Stock codes")
    args = parser.parse_args()

    for stock_code in args.stock_codes:
        get_stock_prices(stock_code)
        # daily_to_weekly(stock_code)
        print("got {} stock prices".format(stock_code))
