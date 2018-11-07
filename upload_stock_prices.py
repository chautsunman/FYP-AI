import argparse
import json

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import pandas as pd

def get_stock_prices(stock_code):
    stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv")

    return {"stockPrices": stock_prices.loc[:, ["timestamp", "adjusted_close"]].values.tolist()}

def upload_stock_prices(stock_code):
    # initialize Firebase admin
    cred = credentials.Certificate("credentials/firebase-adminsdk.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "cmms-fyp.appspot.com"
    })

    bucket = storage.bucket()

    # get the stock prices
    stock_prices = get_stock_prices(stock_code)

    # upload the stock prices to cloud
    stock_prices_json_str = json.dumps(stock_prices)
    blob = bucket.blob("stock_prices/" + stock_code + ".json")
    blob.upload_from_string(stock_prices_json_str, "application/json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload stock prices.")
    parser.add_argument("stock_code", help="Stock code")

    args = parser.parse_args()

    upload_stock_prices(args.stock_code)
