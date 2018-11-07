import json
from os import path, makedirs
import pickle
from hashlib import sha256

class Model:
    SKLEARN_MODEL = "sklearn"

    def __init__(self, model_options):
        self.model = None
        self.model_options = model_options

    def train(self):
        return

    def predict(self):
        return None

    def save(self, saved_model_dir):
        return

    def save_model(self, model_path, model_type):
        if model_type == self.SKLEARN_MODEL:
            with open(model_path, "wb") as model_file:
                pickle.dump(self.model, model_file)

    def load_model(self, model_path, model_type):
        if model_type == self.SKLEARN_MODEL:
            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)

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
    def create_model_dir(self, model_dir_path):
        if not path.isdir(model_dir_path):
            makedirs(model_dir_path)

    @staticmethod
    def get_json_str(d):
        return json.dumps(d)

    @staticmethod
    def hash_str(s):
        return sha256(s.encode()).hexdigest()
