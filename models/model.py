import json
from os import path, makedirs
import pickle
from hashlib import sha256

from keras.models import load_model

class Model:
    SKLEARN_MODEL = "sklearn"
    KERAS_MODEL = "keras"

    def __init__(self, model_options, stock_code=None):
        self.model = None
        self.model_options = model_options
        self.stock_code = stock_code

    def train(self):
        return

    def predict(self):
        return None

    def save(self, saved_model_dir):
        return

    def save_model(self, model_path, model_type):
        self.create_model_dir(path.dirname(model_path))

        if model_type == self.SKLEARN_MODEL:
            with open(model_path, "wb") as model_file:
                pickle.dump(self.model, model_file)
        elif model_type == self.KERAS_MODEL:
            self.model.save(model_path)

    def load_model(self, model_path, model_type):
        if model_type == self.SKLEARN_MODEL:
            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
        elif model_type == self.KERAS_MODEL:
            self.model = load_model(model_path)

    @staticmethod
    def save_models_data(models_data, saved_model_dir):
        with open(path.join(saved_model_dir, "models_data.json"), "w") as models_data_file:
            json.dump(models_data, models_data_file, indent=4)

    @staticmethod
    def load_models_data(saved_model_dir):
        if path.isfile(path.join(saved_model_dir, "models_data.json")):
            with open(path.join(saved_model_dir, "models_data.json"), "r") as models_data_file:
                return json.load(models_data_file)
        else:
            return None

    @staticmethod
    def create_model_dir(model_dir_path):
        if not path.isdir(model_dir_path):
            makedirs(model_dir_path)

    @staticmethod
    def get_json_str(d):
        return json.dumps(d)

    @staticmethod
    def hash_str(s):
        return sha256(s.encode()).hexdigest()
