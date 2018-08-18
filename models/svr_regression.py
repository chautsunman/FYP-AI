import pickle
import json
import time
import os

from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error

from models.model import Model

class SupportVectorRegression(Model):
    def __init__(self, model_options, load=False, saved_model_dir=None):
        Model.__init__(self, model_options)

        print(saved_model_dir)

        # Please check scipy SVR documentation for details
        if not load or saved_model_dir is None:
            self.model = SVR(
                kernel=self.model_options["kernel"], 
                degree=self.model_options["degree"], 
                gamma=self.model_options["gamma"],
                coef0=self.model_options["coef0"],
                tol=self.model_options["tol"],
                C=self.model_options["C"],
                epsilon=self.model_options["epsilon"],
                shrinking=self.model_options["shrinking"],
                cache_size=self.model_options["cache_size"],
                verbose=self.model_options["verbose"],
                max_iter=self.model_options["max_iter"]
            )
        else:
            model_path = self.get_saved_model_path(saved_model_dir)
            if model_path is not None:
                with open(saved_model_dir + "/" + model_path, "rb") as model_file:
                    self.model = pickle.load(model_file)

    def train(self, stock_prices):
        x = np.arange(self.model_options["n"]).reshape(-1, 1)

        print(stock_prices)

        y = stock_prices["change" if not self.model_options["use_stock_price"] else "adjusted_close"]
        # 1-D array is expected
        y = np.flipud(y.values)

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
        Model.create_model_dir(self, saved_model_dir + "/" + self.model_options["stock_code"])

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
        model_type.append(str(self.model_options["kernel"]))
        model_type.append("degree" + str(self.model_options["degree"]))
        model_type.append("C" + str(self.model_options["C"]))
        model_type.append("gamma" + str(self.model_options["gamma"]))
        model_type.append("coef0" + str(self.model_options["coef0"]))
        model_type.append("tol" + str(self.model_options["tol"]))
        model_type.append("epsilon" + str(self.model_options["epsilon"]))
        model_type.append("shrinking" + str(self.model_options["shrinking"]))
        model_type.append("cache_size" + str(self.model_options["cache_size"]))
        model_type.append("verbose" + str(self.model_options["verbose"]))
        model_type.append("max_iter" + str(self.model_options["max_iter"]))
        model_type.append("change" if not self.model_options["use_stock_price"] else "price")
        return "_".join(model_type)

    def get_model_name(self):
        model_name = []
        model_name.append(self.get_model_type())
        model_name.append(str(int(time.time())))
        return "_".join(model_name) + ".model"

    def get_saved_model_path(self, saved_model_dir):
        if not os.path.isfile(saved_model_dir + "/" + "models_data.json"):
            print("No models_data.json")
            return None

        with open(saved_model_dir + "/" + "models_data.json", "r") as models_data_file:
            models_data = json.load(models_data_file)

        if self.model_options["stock_code"] not in models_data["models"]:
            print("No stock code")
            return None

        model_type = self.get_model_type()

        print(models_data)
        print(model_type)

        if model_type not in models_data["models"][self.model_options["stock_code"]]:
            print("No model type")
            return None

        return models_data["models"][self.model_options["stock_code"]][model_type][-1]["model_path"]

    def error(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
