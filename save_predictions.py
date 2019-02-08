import argparse
from datetime import date
import json
import os
import glob

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import numpy as np
import pandas as pd

from models.linear_regression import LinearRegression
from models.svr import SupportVectorRegression
from models.linear_index_regression import LinearIndexRegression
from models.svr_index_regression import SupportVectorIndexRegression
from models.dnn_regression import DenseNeuralNetwork

from build_dataset import build_dataset

from train_models import SAVED_MODELS_DIR_MAP

def get_saved_predictions(stock_code, location="local"):
    """Gets saved predictions directly

    Args:
        stock_code: Stock code specifying a stock.
        location: local or cloud, for now only local has been implemented

    Returns:
        A dict with all predictions and models information.

        Format:
        {
            "predictions": [
                [p11, p12, ...],
                [p21, p22, ...],
                ...
            ],
            "models": [m1_info, m2_info, ...]
        }
    """

    if not os.path.isdir("./saved_predictions/" + stock_code):
        return {}

    else:
        list_of_predictions = glob.glob("./saved_predictions/" + stock_code + "/*.json")
        latest_prediction = max(list_of_predictions)
        with open(latest_prediction) as prediction:
            return json.load(prediction)

def get_predictions(stock_code):
    """Gets the predictions of a stock from all trained models.

    1. Get all saved models.
    2. Build the predict data based on the model's input options.
    3. Predict stock price.

    Args:
        stock_code: Stock code specifying a stock.

    Returns:
        A dict with all predictions and models information.

        Format:
        {
            "predictions": [
                [p11, p12, ...],
                [p21, p22, ...],
                ...
            ],
            "models": [m1_info, m2_info, ...]
        }
    """

    predictions_all = []
    models_all = []

    # get all predictions and models data
    models = LinearRegression.get_all_models(stock_code, SAVED_MODELS_DIR_MAP[LinearRegression.MODEL]) or []
    predictions = []
    for model in models:
        x = build_dataset(model.input_options, model.model_options["predict_n"], False)
        predictions.append(model.predict(x))
    predictions_all += predictions
    models_all += [{"modelName": model.get_model_display_name()} for model in models]
    models = SupportVectorRegression.get_all_models(stock_code, SAVED_MODELS_DIR_MAP[SupportVectorRegression.MODEL]) or []
    predictions = []
    for model in models:
        x = build_dataset(model.input_options, model.model_options["predict_n"], False)
        predictions.append(model.predict(x))
    predictions_all += predictions
    models_all += [{"modelName": model.get_model_display_name()} for model in models]
    models = LinearIndexRegression.get_all_models(stock_code, SAVED_MODELS_DIR_MAP[LinearIndexRegression.MODEL]) or []
    predictions = []
    for model in models:
        x = build_dataset(model.input_options, model.model_options["predict_n"], False)
        predictions.append(model.predict(x))
    predictions_all += predictions
    models_all += [{"modelName": model.get_model_display_name()} for model in models]
    models = SupportVectorIndexRegression.get_all_models(stock_code, SAVED_MODELS_DIR_MAP[SupportVectorIndexRegression.MODEL]) or []
    predictions = []
    for model in models:
        x = build_dataset(model.input_options, model.model_options["predict_n"], False)
        predictions.append(model.predict(x))
    predictions_all += predictions
    models_all += [{"modelName": model.get_model_display_name()} for model in models]
    models = DenseNeuralNetwork.get_all_models(stock_code, SAVED_MODELS_DIR_MAP[DenseNeuralNetwork.MODEL]) or []
    predictions = []
    for model in models:
        x = build_dataset(model.input_options, model.model_options["predict_n"], False)
        predictions.append(model.predict(x))
    predictions_all += predictions
    models_all += [
        {
            "modelName": model.get_model_display_name(),
            "model": "dnn",
            "modelOptions": model.model_options,
            "inputOptions": model.input_options
        }
        for model in models
    ]

    predictions_all = [prediction.tolist() for prediction in predictions_all]

    return {"predictions": predictions_all, "models": models_all}

def save_predictions_local(stock_code):
    """Saves predictions in local in saved_predictions/<stock_code>."""

    # get the predictions
    predictions = get_predictions(stock_code)

    # create the predictions folder for the stock if it does not exist
    if not os.path.isdir("./saved_predictions/" + stock_code):
        os.makedirs("./saved_predictions/" + stock_code)

    # save the predictions and models
    with open("./saved_predictions/" + stock_code + "/" + date.today().isoformat() + ".json", "w") as predictions_file:
        json.dump(predictions, predictions_file, indent=4)

def save_predictions_cloud(stock_code):
    """Saves predictions onto Firebase Cloud Storage in predictions/<stock_code>."""

    # initialize Firebase admin
    cred = credentials.Certificate("credentials/firebase-adminsdk.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "cmms-fyp.appspot.com"
    })

    bucket = storage.bucket()

    # get the predictions
    predictions = get_predictions(stock_code)

    # upload the predictions and models to cloud
    predictions_json_str = json.dumps(predictions)
    blob = bucket.blob("predictions/" + stock_code + "/predictions.json")
    blob.upload_from_string(predictions_json_str, "application/json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save predictions.")
    parser.add_argument("storage", choices=["local", "cloud"], help="Storage")
    parser.add_argument("stock_code", help="Stock code")

    args = parser.parse_args()

    if args.storage == "local":
        save_predictions_local(args.stock_code)
    elif args.storage == "cloud":
        save_predictions_cloud(args.stock_code)
