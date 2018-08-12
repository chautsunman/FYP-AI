from flask import Flask
from flask import request
from flask import jsonify

from flask_cors import cross_origin

import pandas as pd

from models.linear_regression import LinearRegression

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
