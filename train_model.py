import argparse
import os

import pandas as pd

from models.linear_regression import LinearRegression
from models.svr_regression import SupportVectorRegression

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
    elif args.model == "svr":
        model = SupportVectorRegression({
            "stock_code": args.regression_stock_code,
            "use_stock_price": args.regression_use_stock_price,
            "n": args.regression_n,
            "kernel": args.kernel,
            "degree": args.degree,
            "gamma": args.gamma if args.gamma != -1 else "auto",
            "coef0": args.coef0,
            "tol": args.tol,
            "C": args.C,
            "epsilon": args.epsilon,
            "shrinking": args.shrinking,
            "cache_size": args.cache_size,
            "verbose": args.verbose,
            "max_iter": args.max_iter
        })

        stock_prices = pd.read_csv("./data/stock_prices/" + args.regression_stock_code + ".csv", nrows=args.regression_n)

        model.train(stock_prices)

        model.save("./saved_models/svr")
    else:
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("model", choices=["linear", "svr"], help="Model")
    parser.add_argument("--regression-stock-code", help="Stock code")
    parser.add_argument("--regression-use-stock-price", action="store_true", help="Use stock price as the data")
    parser.add_argument("--regression-n", default=30, type=int, choices=[30, 90, 180, 365], help="Number of latest stock prices to use for regression")

    # For SVR, check the SVR scipy documentation for details
    parser.add_argument("--C", default=1.0, type=float, help="Penalty parameter C of the error term")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Epsilon in the epsilon-SVR model")
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly", "sigmoid"], help="Epsilon in the epsilon-SVR model")
    parser.add_argument("--degree", default=3, type=int, help="Degree of the polynomial kernel function ('poly').  Ignored by other kernels")
    parser.add_argument("--gamma", default=-1, type=float, help="Kernel coefficient for 'rbf', 'sigmoid' and 'poly', defaulted to 'auto', which uses 1/n_features")
    parser.add_argument("--coef0", default=0.0, type=float, help="Independent term in kernel function, used by 'poly' and 'sigmoid'")
    parser.add_argument("--shrinking", default=True, type=bool, help="Whether to use shrinking heuristic")
    parser.add_argument("--tol", default=1e-3, type=float, help="Tolerance for stopping criterion")
    parser.add_argument("--cache-size", default=500.0, type=float, help="Size of kernel cache in MB")
    parser.add_argument("--verbose", default=False, type=bool, help="Enable verbose output")
    parser.add_argument("--max_iter", default=-1, type=int, help="Max iterations within solver, -1 for no limit")

    args = parser.parse_args()

    train_model(args)
