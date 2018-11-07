import argparse
import json
import os

import pandas as pd

from models.linear_regression import LinearRegression
from models.svr import SupportVectorRegression
from models.linear_index_regression import LinearIndexRegression
from models.svr_index_regression import SupportVectorIndexRegression
from models.dnn_regression import DenseNeuralNetwork

def train_models(train_models_data):
    if not os.path.isdir("./saved_models"):
        os.makedirs("./saved_models")

    for train_model_data in train_models_data:
        if train_model_data["model"] == LinearRegression.MODEL:
            model = LinearRegression(train_model_data["modelOptions"])

        elif train_model_data["model"] == SupportVectorRegression.MODEL:
            model = SupportVectorRegression(train_model_data["modelOptions"])

        elif train_model_data["model"] == LinearIndexRegression.MODEL:
            model = LinearIndexRegression(train_model_data["modelOptions"])

            stock_prices = pd.read_csv("./data/stock_prices/" + train_model_data["modelOptions"]["stock_code"] + ".csv", nrows=train_model_data["modelOptions"]["n"])

            model.train(stock_prices)

            model.save("./saved_models/linear_index_regression")

        elif train_model_data["model"] == SupportVectorIndexRegression.MODEL:
            model = SupportVectorIndexRegression(train_model_data["modelOptions"])

            stock_prices = pd.read_csv("./data/stock_prices/" + train_model_data["modelOptions"]["stock_code"] + ".csv", nrows=train_model_data["modelOptions"]["n"])

            model.train(stock_prices)

            model.save("./saved_models/svr_index_regression")

        elif train_model_data["model"] == DenseNeuralNetwork.MODEL:
            model = DenseNeuralNetwork(train_model_data)

            stock_prices = pd.read_csv("./data/stock_prices/" + train_model_data["modelOptions"]["stock_code"] + ".csv", nrows=train_model_data["modelOptions"]["n"])

            model.train(stock_prices)

            model.save("./saved_models/dnn")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models.")
    parser.add_argument("train_models_json", help="Train models JSON file path")

    args = parser.parse_args()

    with open('./' + args.train_models_json) as train_models_json_file:
        train_models_data = json.load(train_models_json_file)

    train_models(train_models_data['models'])
