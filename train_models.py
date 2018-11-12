import argparse
import json
from os import path, makedirs

import pandas as pd

from models.linear_regression import LinearRegression
from models.svr import SupportVectorRegression
from models.linear_index_regression import LinearIndexRegression
from models.svr_index_regression import SupportVectorIndexRegression
from models.dnn_regression import DenseNeuralNetwork

from build_dataset import build_dataset

SAVED_MODELS_DIR = path.join(".", "saved_models")
SAVED_MODELS_DIR_MAP = {
    LinearRegression.MODEL: path.join(SAVED_MODELS_DIR, "linear_regression"),
    SupportVectorRegression.MODEL: path.join(SAVED_MODELS_DIR, "svr"),
    LinearIndexRegression.MODEL: path.join(SAVED_MODELS_DIR, "linear_index_regression"),
    SupportVectorIndexRegression.MODEL: path.join(SAVED_MODELS_DIR, "svr_index_regression"),
    DenseNeuralNetwork.MODEL: path.join(SAVED_MODELS_DIR, "dnn")
}

def train_models(train_models_data):
    if not path.isdir(SAVED_MODELS_DIR):
        makedirs(SAVED_MODELS_DIR)

    for train_model_data in train_models_data:
        # initialize the model
        if train_model_data["model"] == LinearRegression.MODEL:
            model = LinearRegression(train_model_data["modelOptions"], stock_code=train_model_data["stockCode"])
        elif train_model_data["model"] == SupportVectorRegression.MODEL:
            model = SupportVectorRegression(train_model_data["modelOptions"], stock_code=train_model_data["stockCode"])
        elif train_model_data["model"] == LinearIndexRegression.MODEL:
            model = LinearIndexRegression(train_model_data["modelOptions"], train_model_data["stock_code"])
        elif train_model_data["model"] == SupportVectorIndexRegression.MODEL:
            model = SupportVectorIndexRegression(train_model_data["modelOptions"], train_model_data["stock_code"])
        elif train_model_data["model"] == DenseNeuralNetwork.MODEL:
            model = DenseNeuralNetwork(train_model_data["modelOptions"], stock_code=train_model_data["stockCode"])

        # prepare the data and train the model
        x, y = build_dataset(train_model_data["inputOptions"], True)
        model.train(x, y)

        # save the model
        model.save(SAVED_MODELS_DIR_MAP[train_model_data["model"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models.")
    parser.add_argument("train_models_json", help="Train models JSON file path")

    args = parser.parse_args()

    with open('./' + args.train_models_json) as train_models_json_file:
        train_models_data = json.load(train_models_json_file)

    train_models(train_models_data['models'])
