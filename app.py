from datetime import date
import json
import os

from flask import Flask
from flask import request
from flask import jsonify

from flask_cors import cross_origin

import pandas as pd

from upload_stock_prices import get_stock_prices
from save_predictions import get_predictions
from models.dnn_regression import DenseNeuralNetwork

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello world."

@app.route("/stockPrices/<stock_code>")
@cross_origin({
    "origins": ["localhost"],
    "methods": "GET"
})
def stock_prices(stock_code):
    stock_prices = get_stock_prices(stock_code)
    return jsonify(stock_prices)

@app.route("/predict/<stock_code>")
@cross_origin({
    "origins": ["localhost"],
    "methods": "GET"
})
def predict(stock_code):
    predictions = get_predictions(stock_code)

    return jsonify({"success": True, "predictions": predictions["predictions"], "models": predictions["models"]})

@app.route("/model/dnn/predict/<stock_code>")
@cross_origin({
    "origins": ["localhost"],
    "methods": "GET"
})
def dnn_predict(stock_code):
    if "useStockPrice" not in request.args or "n" not in request.args:
        return jsonify({"success": False, "error": {"code": "invalid-argument"}})

    model_options = {
        "stock_code": stock_code,
        "use_stock_price": False if request.args.get("useStockPrice") != "true" else True,
        "n": int(request.args.get("n")),
        "lookback": int(request.args.get("lookback"))
    }

    model = DenseNeuralNetwork(model_options, load=True, saved_model_dir="./saved_models/dnn")
    if model.model is None:
        return jsonify({"success": False, "error": {"code": "invalid-argument"}})

    if not model_options["use_stock_price"]:
        stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv", nrows=int(request.args.get("lookback")))
        print(stock_prices.loc[:, "adjusted_close"])
        predictions = model.predict(stock_prices.loc[:, "adjusted_close"])
    else:
        print(model)
        stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv", nrows=int(request.args.get("lookback")))
        predictions = model.predict(stock_prices.loc[:, "adjusted_close"])

    return jsonify({"success": True, "predictions": predictions.tolist()})

