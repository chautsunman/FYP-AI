from datetime import date
import json
import os

from flask import Flask
from flask import request
from flask import jsonify

from flask_cors import cross_origin

import pandas as pd

from models.linear_regression import LinearRegression, get_all_predictions as get_all_linear_predictions
from models.svr_regression import SupportVectorRegression, get_all_predictions as get_all_svr_predictions
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
def get_stock_prices(stock_code):
    stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv")
    return jsonify({"stockPriceData": stock_prices.loc[:, ["timestamp", "adjusted_close"]].values.tolist()})

@app.route("/predict/<stock_code>")
@cross_origin({
    "origins": ["localhost"],
    "methods": "GET"
})
def predict(stock_code):
    stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv", nrows=1)

    predictions_all = []

    predictions_linear, models_linear = get_all_linear_predictions(stock_code, "./saved_models/linear", stock_prices.loc[0, "adjusted_close"])
    predictions_svr, models_svr = get_all_svr_predictions(stock_code, "./saved_models/svr", stock_prices.loc[0, "adjusted_close"])

    predictions_all = predictions_linear + predictions_svr
    models_all = models_linear + models_svr

    predictions_all = [prediction.tolist() for prediction in predictions_all]
    models_all = [{"modelName": model.get_model_display_name()} for model in models_all]

    return jsonify({"success": True, "predictions": predictions_all, "models": models_all})

@app.route("/save_predictions/<stock_code>")
def save_predictions(stock_code):
    stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv", nrows=1)

    # get all predictions and models
    predictions, models = get_all_linear_predictions(stock_code, "./saved_models/linear", stock_prices.loc[0, "adjusted_close"])

    # format predictions
    predictions = [prediction.tolist() for prediction in predictions]

    # format models
    models = [{"modelName": model.get_model_display_name()} for model in models]

    # create the predictions folder for the stock if it does not exist
    if not os.path.isdir("./saved_predictions/" + stock_code):
        os.makedirs("./saved_predictions/" + stock_code)

    # save the predictions and models
    with open("./saved_predictions/" + stock_code + "/" + date.today().isoformat() + ".json", "w") as predictions_file:
        json.dump({"predictions": predictions, "models": models}, predictions_file)

    return jsonify({"success": True})

@app.route("/model/linear/predict/<stock_code>")
@cross_origin({
    "origins": ["localhost"],
    "methods": "GET"
})
def linear_predict(stock_code):
    if "useStockPrice" not in request.args or "n" not in request.args:
        return jsonify({"success": False, "error": {"code": "invalid-argument"}})

    model_options = {
        "stock_code": stock_code,
        "use_stock_price": False if request.args.get("useStockPrice") != "true" else True,
        "n": int(request.args.get("n"))
    }

    model = LinearRegression(model_options, load=True, saved_model_dir="./saved_models/linear")
    if model.model is None:
        return jsonify({"success": False, "error": {"code": "invalid-argument"}})

    if not model_options["use_stock_price"]:
        stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv", nrows=1)
        predictions = model.predict(stock_prices.loc[0, "adjusted_close"])
    else:
        predictions = model.predict()

    return jsonify({"success": True, "predictions": predictions.tolist()})

@app.route("/model/svr/predict/<stock_code>")
@cross_origin({
    "origins": ["localhost"],
    "methods": "GET"
})
def svr_predict(stock_code):
    if "useStockPrice" not in request.args or "n" not in request.args:
        return jsonify({"success": False, "error": {"code": "invalid-argument"}})

    model_options = {
        "stock_code": stock_code,
        "use_stock_price": False if request.args.get("useStockPrice") != "true" else True,
        "n": int(request.args.get("n")),
        "kernel": request.args.get("kernel") or "rbf",
        "C": float(request.args.get("C") or 1.0),
        "epsilon": float(request.args.get("epsilon") or 0.1),
        "degree": int(request.args.get("degree") or 3),
        "gamma": int(request.args.get("gamma")) if request.args.get("gamma") is not None else "auto",
        "coef0": float(request.args.get("coef0") or 0.0),
        "tol": float(request.args.get("tol") or 0.001),
        "shrinking": bool(request.args.get("shrinking") or True),
        "cache_size": float(request.args.get("cache_size") or 500.0),
        "verbose": bool(request.args.get("verbose") or False),
        "max_iter": int(request.args.get("max_iter") or -1)
    }

    model = SupportVectorRegression(model_options, load=True, saved_model_dir="./saved_models/svr")
    if model.model is None:
        return jsonify({"success": False, "error": {"code": "invalid-argument"}})

    if not model_options["use_stock_price"]:
        stock_prices = pd.read_csv("./data/stock_prices/" + stock_code + ".csv", nrows=1)
        predictions = model.predict(stock_prices.loc[0, "adjusted_close"])
    else:
        print(model)
        predictions = model.predict()

    return jsonify({"success": True, "predictions": predictions.tolist()})

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

