import argparse
import json
import os

import pandas as pd

from models.linear_regression import LinearRegression
from models.svr_regression import SupportVectorRegression
from models.dnn_regression import DenseNeuralNetwork

def train_models(model_options):
    if not os.path.isdir("./saved_models"):
        os.makedirs("./saved_models")

    for model_option in model_options:
        if model_option["model"] == "linear":
            model = LinearRegression(model_option)

            stock_prices = pd.read_csv("./data/stock_prices/" + model_option["stock_code"] + ".csv", nrows=model_option["n"])

            model.train(stock_prices)

            model.save("./saved_models/linear")

        elif model_option["model"] == "svr":
            model = SupportVectorRegression(model_option)

            stock_prices = pd.read_csv("./data/stock_prices/" + model_option["stock_code"] + ".csv", nrows=model_option["n"])

            model.train(stock_prices)

            model.save("./saved_models/svr")

        elif model_option["model"] == "dnn":
            model = DenseNeuralNetwork(model_option)

            stock_prices = pd.read_csv("./data/stock_prices/" + model_option["stock_code"] + ".csv", nrows=model_option["n"])

            model.train(stock_prices)

            model.save("./saved_models/dnn")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models.")
    parser.add_argument("train_models_json", help="Train models JSON file path")

    args = parser.parse_args()

    with open('./' + args.train_models_json) as train_models_json_file:
        model_options = json.load(train_models_json_file)

    train_models(model_options['models'])
