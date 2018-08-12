import argparse
import os

import pandas as pd

from models.linear_regression import LinearRegression

def train_model(args):
    if not os.path.isdir("./saved_models"):
        os.makedirs("./saved_models")

    if args.model == "linear":
        model = LinearRegression({
            "stock_code": args.regression_stock_code,
            "use_stock_price": args.regression_use_stock_price,
            "n": args.regression_n
        })

        stock_prices = pd.read_csv("./data/stock_prices/" + args.regression_stock_code + ".csv", nrows=args.regression_n)

        model.train(stock_prices)

        model.save("./saved_models/linear")
    else:
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("model", choices=["linear"], help="Model")
    parser.add_argument("--regression-stock-code", help="Stock code")
    parser.add_argument("--regression-use-stock-price", action="store_true", help="Use stock price as the data")
    parser.add_argument("--regression-n", default=30, type=int, choices=[30, 90, 180, 365], help="Number of latest stock prices to use for regression")
    args = parser.parse_args()

    train_model(args)
