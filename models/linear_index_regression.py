import pickle
import json
import time
import os

from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error

from models.index_regression import IndexRegressionModel

class LinearIndexRegression(IndexRegressionModel):
    MODEL = "linear_index_regression"

    def __init__(self, model_options, load=False, saved_model_dir=None, saved_model_path=None):
        IndexRegressionModel.__init__(self, model_options)

        if not load or saved_model_dir is None:
            self.model = linear_model.LinearRegression()
        else:
            model_path = saved_model_path if saved_model_path is not None else self.get_saved_model_path(saved_model_dir)
            if model_path is not None:
                with open(saved_model_dir + "/" + model_path, "rb") as model_file:
                    self.model = pickle.load(model_file)

    def train(self, stock_prices):
        x = np.arange(self.model_options["n"]).reshape(-1, 1)

        y = stock_prices["change" if not self.model_options["use_stock_price"] else "adjusted_close"]
        y = np.flipud(y.values.reshape(-1, 1))

        self.model.fit(x, y)

    def predict(self, last_price=None):
        if not self.model_options["use_stock_price"] and last_price is None:
            return None

        x = np.arange(self.model_options["n"], self.model_options["n"] + 30).reshape(-1, 1)

        predictions = self.model.predict(x).flatten()

        if not self.model_options["use_stock_price"]:
            predictions[0] = last_price * (1 + predictions[0])
            for i in range(1, predictions.shape[0]):
                predictions[i] = predictions[i - 1] * (1 + predictions[i])

        return predictions

    def save(self, saved_model_dir):
        self.create_model_dir(self, saved_model_dir + "/" + self.model_options["stock_code"])

        model_name = self.get_model_name()
        model_path = self.model_options["stock_code"] + "/" + model_name

        with open(saved_model_dir + "/" + model_path, "wb") as model_file:
            pickle.dump(self.model, model_file)

        if os.path.isfile(saved_model_dir + "/" + "models_data.json"):
            with open(saved_model_dir + "/" + "models_data.json", "r") as models_data_file:
                models_data = json.load(models_data_file)
        else:
            models_data = {"models": {}}

        if self.model_options["stock_code"] not in models_data["models"]:
            models_data["models"][self.model_options["stock_code"]] = {}

        model_type = self.get_model_type()

        if model_type not in models_data["models"][self.model_options["stock_code"]]:
            models_data["models"][self.model_options["stock_code"]][model_type] = []

        model_data = self.model_options
        model_data["model_name"] = model_name
        model_data["model_path"] = model_path

        models_data["models"][self.model_options["stock_code"]][model_type].append(model_data)

        with open(saved_model_dir + "/" + "models_data.json", "w") as models_data_file:
            json.dump(models_data, models_data_file)

    def get_model_type(self):
        model_type = []
        model_type.append(str(self.model_options["n"]) + "days")
        model_type.append("change" if not self.model_options["use_stock_price"] else "price")
        return "_".join(model_type)

    def get_model_name(self):
        model_name = []
        model_name.append(self.get_model_type())
        model_name.append(str(int(time.time())))
        return "_".join(model_name) + ".model"

    def get_saved_model_path(self, saved_model_dir):
        if not os.path.isfile(saved_model_dir + "/" + "models_data.json"):
            return None

        with open(saved_model_dir + "/" + "models_data.json", "r") as models_data_file:
            models_data = json.load(models_data_file)

        if self.model_options["stock_code"] not in models_data["models"]:
            return None

        model_type = self.get_model_type()

        if model_type not in models_data["models"][self.model_options["stock_code"]]:
            return None

        return models_data["models"][self.model_options["stock_code"]][model_type][-1]["model_path"]

    def get_model_display_name(self):
        options_name = [str(self.model_options["n"]), "days", "change" if not self.model_options["use_stock_price"] else "price"]
        return "Linear Regression (%s)" % " ".join(options_name)

    def error(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

def get_all_predictions(stock_code, saved_model_dir, last_price):
    with open(saved_model_dir + "/models_data.json", "r") as models_data_file:
        models_data = json.load(models_data_file)

    if stock_code not in models_data["models"]:
        return [], []

    models_data = models_data["models"][stock_code]

    models = []
    for _, model_options in models_data.items():
        models.append(LinearIndexRegression(model_options[-1], load=True, saved_model_dir=saved_model_dir, saved_model_path=model_options[-1]["model_path"]))

    predictions = []
    for model in models:
        if not model.model_options["use_stock_price"]:
            predictions.append(model.predict(last_price))
        else:
            predictions.append(model.predict())

    return predictions, models
