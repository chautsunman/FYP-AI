import argparse
import json
from os import path, makedirs

import pandas as pd

from models.linear_regression import LinearRegression
from models.svr import SupportVectorRegression
from models.linear_index_regression import LinearIndexRegression
from models.svr_index_regression import SupportVectorIndexRegression
from models.dnn_regression import DenseNeuralNetwork

from build_dataset import build_training_dataset

SAVED_MODELS_DIR = path.join(".", "saved_models")
SAVED_MODELS_DIR_MAP = {
    LinearRegression.MODEL: path.join(SAVED_MODELS_DIR, "linear_regression"),
    SupportVectorRegression.MODEL: path.join(SAVED_MODELS_DIR, "svr"),
    LinearIndexRegression.MODEL: path.join(SAVED_MODELS_DIR, "linear_index_regression"),
    SupportVectorIndexRegression.MODEL: path.join(SAVED_MODELS_DIR, "svr_index_regression"),
    DenseNeuralNetwork.MODEL: path.join(SAVED_MODELS_DIR, "dnn")
}

def train_models(train_models_data):
    """Trains models.

    Args:
        train_models_data: Train models data.
            Format:
            {
                models: [
                    {
                        "model": "model type, matches MODEL in a model class",
                        "stockCode": "the predicting stock",
                        "modelOptions": "model options dict",
                        "inputOptions": "input options dict"
                    }
                ]
            }
            Refer to train_models_sample.json.

    """

    if not path.isdir(SAVED_MODELS_DIR):
        makedirs(SAVED_MODELS_DIR)

    for train_model_data_idx, train_model_data in enumerate(train_models_data):
        print("Model {}".format(train_model_data_idx + 1))

        # initialize the model
        if train_model_data["model"] == LinearRegression.MODEL:
            model = LinearRegression(train_model_data["modelOptions"], train_model_data["inputOptions"], stock_code=train_model_data["stockCode"])
        elif train_model_data["model"] == SupportVectorRegression.MODEL:
            model = SupportVectorRegression(train_model_data["modelOptions"], train_model_data["inputOptions"], stock_code=train_model_data["stockCode"])
        elif train_model_data["model"] == LinearIndexRegression.MODEL:
            model = LinearIndexRegression(train_model_data["modelOptions"], train_model_data["inputOptions"], train_model_data["stock_code"])
        elif train_model_data["model"] == SupportVectorIndexRegression.MODEL:
            model = SupportVectorIndexRegression(train_model_data["modelOptions"], train_model_data["inputOptions"], train_model_data["stock_code"])
        elif train_model_data["model"] == DenseNeuralNetwork.MODEL:
            model = DenseNeuralNetwork(train_model_data["modelOptions"], train_model_data["inputOptions"], stock_code=train_model_data["stockCode"])

        # prepare the data
        x, y, other_data = build_training_dataset(train_model_data["inputOptions"], model.model_options["predict_n"])
        if train_model_data["model"] in [LinearRegression.MODEL, SupportVectorRegression.MODEL, DenseNeuralNetwork.MODEL]:
            # get the training set
            x = x[:-100]
            y = y[:-100]
        if "normalize" in train_model_data["inputOptions"]:
            model.input_options["normalize_data"] = other_data["normalize_data"]
        # train the model
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
