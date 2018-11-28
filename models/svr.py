from os import path
import time

from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error

from models.model import Model

class SupportVectorRegression(Model):
    """Support vector regression model."""

    MODEL = "svr"

    def __init__(self, model_options, input_options, stock_code=None, load=False, saved_model_dir=None, saved_model_path=None):
        """Initializes the model. Creates a new model or loads a saved model."""

        Model.__init__(self, model_options, input_options, stock_code=stock_code)

        # Please check scipy SVR documentation for details
        if not load or saved_model_dir is None:
            self.model = [SVR(
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
            ) for _ in range(model_options["predict_n"])]
        else:
            model_path = saved_model_path if saved_model_path is not None else self.get_saved_model_path(saved_model_dir)
            if model_path is not None:
                self.load_model(path.join(saved_model_dir, model_path), self.SKLEARN_MODEL)

    def train(self, xs, ys):
        """Trains the model.

        Args:
            xs: A m-by-n NumPy data array of m data with n features.
            ys: A Numpy label array of m data.
        """

        for i in range(self.model_options["predict_n"]):
            self.model[i].fit(xs, ys[:, i])

    def predict(self, x):
        """Predicts.

        Returns:
            A NumPy array of the prediction.
        """

        return np.array([model.predict(x).flatten() for model in self.model]).flatten()

    def save(self, saved_model_dir):
        """Saves the model in saved_model_dir.

        1. Saves the model.
        2. Saves the models data.

        Directory structure:
            <saved_model_dir>
                model_type_hash
                    stock_code
                        (models)
                models_data.json

        Args:
            saved_model_dir: A path to a directory where the model will be saved in.
        """

        # create the saved models directory
        self.create_model_dir(saved_model_dir)

        model_name = self.get_model_name()
        stock_code = "general" if self.stock_code is None else self.stock_code
        model_path = path.join(self.get_model_type_hash(), stock_code)

        # save the model
        self.save_model(path.join(saved_model_dir, model_path, model_name), self.SKLEARN_MODEL_ARRAY)

        # load models data
        models_data = self.load_models_data(saved_model_dir)
        if models_data is None:
            models_data = {"models": {}, "modelTypes": {}}

        # update models data
        models_data = self.update_models_data(models_data, model_name, model_path)

        # save models data
        self.save_models_data(models_data, saved_model_dir)

    def update_models_data(self, models_data, model_name, model_path):
        """Updates models data to include this newly saved model.

        Models data dict format:
        {
            "models": {
                "<model_type_hash1>": {
                    "<stock_code1>": [
                        {"model_name": "model 1 name", "model_path": "model 1 path", "model": "linear_regression"},
                        {"model_name": "model 2 name", "model_path": "model 2 path", "model": "linear_regression"},
                        ...
                    ],
                    "<stock_code2": [...],
                    "general": [],
                    ...
                },
                "<model_type_hash2>": {...}
            },
            "modelTypes": {
                "<model_type_hash1>": <model_type1_dict>,
                "<model_type_hash2>": <model_type2_dict>,
                ...
            }
        }

        Args:
            models_data: Old models data dict.
            model_name: Saved model name.
            model_path: Saved model path.

        Returns:
            Updated models_data dict.
        """

        model_type_hash = self.get_model_type_hash()

        if model_type_hash not in models_data["models"]:
            models_data["models"][model_type_hash] = {"general": []}

        stock_code = "general" if self.stock_code is None else self.stock_code

        if stock_code not in models_data["models"][model_type_hash]:
            models_data["models"][model_type_hash][stock_code] = []

        model_data = {}
        model_data["model_name"] = model_name
        model_data["model_path"] = model_path
        model_data["model"] = self.MODEL

        models_data["models"][model_type_hash][stock_code].append(model_data)

        if model_type_hash not in models_data["modelTypes"]:
            models_data["modelTypes"][model_type_hash] = self.get_model_type()

        return models_data

    def get_model_type(self):
        """Returns model type (model, model options, input options)."""

        return {"model": self.MODEL, "modelOptions": self.model_options, "inputOptions": self.input_options}

    def get_model_type_hash(self):
        """Returns model type hash."""

        model_type = self.get_model_type()

        model_type_json_str = self.get_json_str(model_type)

        return self.hash_str(model_type_json_str)

    def get_model_name(self):
        """Returns model name (<model_type_hash>_<time>.model)."""

        model_name = []
        model_name.append(self.get_model_type_hash())
        model_name.append(str(int(time.time())))
        return "_".join(model_name) + ".model"

    def get_saved_model_path(self, saved_model_dir):
        """Returns model path of the latest saved same type model by searching the models data file, or None if not found."""

        models_data = self.load_models_data(saved_model_dir)
        if models_data is None:
            return None

        model_type_hash = self.get_model_type_hash()

        if model_type_hash not in models_data["models"]:
            return None

        stock_code = "general" if self.stock_code is None else self.stock_code

        if stock_code not in models_data["models"][model_type_hash][stock_code]:
            return None

        return models_data["models"][model_type_hash][stock_code][-1]["model_path"]

    # Return the name of the model in displayable format
    def get_model_display_name(self):
        """Returns model display name for the app."""

        return "SVM Regression, Kernel = {}".format(self.model_options["kernel"])

    def error(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def get_all_models(stock_code, saved_model_dir):
        """Returns an array of all different types saved models by searching the models data file."""

        models_data = Model.load_models_data(saved_model_dir)
        if models_data is None:
            return None

        models = []
        for model_type in models_data["models"]:
            if len(models_data["models"][model_type]["general"]) > 0:
                models.append(SupportVectorRegression(
                    models_data["modelTypes"][model_type]["modelOptions"],
                    models_data["modelTypes"][model_type]["inputOptions"],
                    stock_code=stock_code,
                    load=True,
                    saved_model_dir=saved_model_dir,
                    saved_model_path=path.join(
                        models_data["models"][model_type]["general"][-1]["model_path"],
                        models_data["models"][model_type]["general"][-1]["model_name"])
                ))
            if len(models_data["models"][model_type][stock_code]) > 0:
                models.append(SupportVectorRegression(
                    models_data["modelTypes"][model_type]["modelOptions"],
                    models_data["modelTypes"][model_type]["inputOptions"],
                    stock_code=stock_code,
                    load=True,
                    saved_model_dir=saved_model_dir,
                    saved_model_path=path.join(
                        models_data["models"][model_type][stock_code][-1]["model_path"],
                        models_data["models"][model_type][stock_code][-1]["model_name"])
                ))

        return models
