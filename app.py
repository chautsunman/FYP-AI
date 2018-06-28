from flask import Flask
from flask.json import jsonify

app = Flask(__name__)

import numpy as np
from data.linear_data import get_linear_data

from models.linear_regression import LinearRegression

@app.route("/")
def hello_world():
    return "Hello world."

@app.route("/model/linear_regression/train")
def train_linear_regression():
    x, y = get_linear_data(100)

    model = LinearRegression()

    model.train(x, y)

    return jsonify({"w": model.model.coef_.tolist(), "b": model.model.intercept_})

@app.route("/model/linear_regression/predict")
def linear_regression_predict():
    x = np.random.randn(10, 2)

    model = LinearRegression()
    model.model.coef_ = np.array([2, 16])
    model.model.intercept_ = 18

    return jsonify(model.predict(x).tolist())
