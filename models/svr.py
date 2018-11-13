from os import path
import time

from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error

from models.model import Model

class SupportVectorRegression(Model):
    MODEL = "svr"

    def __init__(self, model_options, load=False, saved_model_dir=None, saved_model_path=None):
        Model.__init__(self, model_options)

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
            model_path = saved_model_path if saved_model_path is not None else self.get_saved_model_path(saved_model_dir)
            if model_path is not None:
                self.load_model(path.join(saved_model_dir, model_path), self.SKLEARN_MODEL)

    def train(self, xs, ys):
        self.model.fit(xs, ys)

    def predict(self, x):
        return self.model.predict([x]).flatten()

    def save(self, saved_model_dir):
        # create the saved models directory
        self.create_model_dir(self, saved_model_dir)

        model_name = self.get_model_name()
        model_path = self.get_model_type_hash()

        # save the model
        self.save_model(path.join(saved_model_dir, model_path, model_name), self.SKLEARN_MODEL)

        # load models data
        models_data = self.load_models_data(saved_model_dir)
        if models_data is None:
            models_data = {"models": []}

        # update models data
        models_data = self.update_models_data(models_data, model_name, model_path)

        # save models data
        self.save_models_data(models_data, saved_model_dir)

    def update_models_data(self, models_data, model_name, model_path):
        model_type_hash = self.get_model_type_hash()

        if model_type_hash not in models_data["models"]:
            models_data["models"][model_type_hash] = []

        model_data = {}
        model_data["model_name"] = model_name
        model_data["model_path"] = model_path
        model_data["model"] = self.MODEL

        models_data["models"][model_type_hash].append(model_data)

        if model_type_hash not in models_data["modelTypes"]:
            models_data["modelTypes"][model_type_hash] = self.get_model_type()

        return models_data

    def get_model_type(self):
        return {"model": self.MODEL, "modelOptions": self.model_options}

    def get_model_type_hash(self):
        model_type = self.get_model_type()

        model_type_json_str = self.get_json_str(model_type)

        return self.hash_str(model_type_json_str)

    def get_model_name(self):
        model_name = []
        model_name.append(self.get_model_type_hash())
        model_name.append(str(int(time.time())))
        return "_".join(model_name) + ".model"

    def get_saved_model_path(self, saved_model_dir):
        models_data = self.load_models_data(saved_model_dir)
        if models_data is None:
            return None

        model_type_hash = self.get_model_type_hash()

        if model_type_hash not in models_data["models"]:
            return None

        return models_data["models"][model_type_hash][-1]["model_path"]

    # Return the name of the model in displayable format
    def get_model_display_name(self):
        return "SVM Regression, Kernel = {} ({} days)".format(self.model_options["kernel"], self.model_options["n"])

    def error(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def get_all_predictions(stock_code, saved_model_dir):
        models_data = Model.load_models_data(saved_model_dir)
        if models_data is None:
            return None

        models = []
        for model_type, model_data in models_data["models"].items():
            models.append(SupportVectorRegression(
                models_data["modelTypes"][model_type]["modelOptions"],
                load=True,
                saved_model_dir=saved_model_dir,
                saved_model_path=model_data[-1]["model_path"]))

        predictions = []
        for model in models:
            predictions.append(np.array([]))

        return predictions, models
