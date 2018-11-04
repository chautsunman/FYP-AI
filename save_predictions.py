import argparse
from datetime import date
import json
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import pandas as pd

from models.linear_regression import get_all_predictions as get_all_linear_predictions
from models.svr_regression import get_all_predictions as get_all_svr_predictions

def get_predictions(stock_code):
    """Get the predictions of a stock from all trained models.

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

    stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv", nrows=1)

    predictions_all = []

    # get all predictions and models
    predictions_linear, models_linear = get_all_linear_predictions(stock_code, "./saved_models/linear", stock_prices.loc[0, "adjusted_close"])
    predictions_svr, models_svr = get_all_svr_predictions(stock_code, "./saved_models/svr", stock_prices.loc[0, "adjusted_close"])

    predictions_all = predictions_linear + predictions_svr
    models_all = models_linear + models_svr

    # format predictions and models
    predictions_all = [prediction.tolist() for prediction in predictions_all]
    models_all = [{"modelName": model.get_model_display_name()} for model in models_all]

    return {"predictions": predictions_all, "models": models_all}

def save_predictions_local(stock_code):
    # get the predictions
    predictions = get_predictions(stock_code)

    # create the predictions folder for the stock if it does not exist
    if not os.path.isdir("./saved_predictions/" + stock_code):
        os.makedirs("./saved_predictions/" + stock_code)

    # save the predictions and models
    with open("./saved_predictions/" + stock_code + "/" + date.today().isoformat() + ".json", "w") as predictions_file:
        json.dump(predictions, predictions_file)

def save_predictions_cloud(stock_code):
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
