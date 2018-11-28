import json
from os import path, makedirs
import pickle
from hashlib import sha256

from keras.models import load_model

class Model:
    """Base class for all models."""

    SKLEARN_MODEL = "sklearn"
    SKLEARN_MODEL_ARRAY = "sklearn_array"
    KERAS_MODEL = "keras"

    def __init__(self, model_options, input_options, stock_code=None):
        """Initializes model, model options, input options and predicting stock."""

        self.model = None
        self.model_options = model_options
        self.input_options = input_options
        self.stock_code = stock_code

    def train(self):
        """Trains the model."""
        return

    def predict(self):
        """Predicts."""
        return None

    def save(self, saved_model_dir):
        """Saves the model in saved_model_dir."""
        return

    def save_model(self, model_path, model_type):
        """Saves the model to the model path.

        Args:
            model_path: Model path.
            model_type: Library that the model is built on top of.
        """

        # create the model directory
        self.create_model_dir(path.dirname(model_path))

        if model_type == self.SKLEARN_MODEL or model_type == self.SKLEARN_MODEL_ARRAY:
            # save the scikit-learn model with pickle
            with open(model_path, "wb") as model_file:
                pickle.dump(self.model, model_file)
        elif model_type == self.KERAS_MODEL:
            # save the Keras model
            self.model.save(model_path)

    def load_model(self, model_path, model_type):
        """Loads the model from the model path.

        Args:
            model_path: Model path.
            model_type: Library that the model is built on top of.
        """

        if model_type == self.SKLEARN_MODEL or model_type == self.SKLEARN_MODEL_ARRAY:
            # load the scikit-learn model with pickle
            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
        elif model_type == self.KERAS_MODEL:
            # load the Keras model
            self.model = load_model(model_path)

    @staticmethod
    def save_models_data(models_data, saved_model_dir):
        """Saves the models data in <saved_model_dir>/models_data.json."""

        with open(path.join(saved_model_dir, "models_data.json"), "w") as models_data_file:
            json.dump(models_data, models_data_file, indent=4)

    @staticmethod
    def load_models_data(saved_model_dir):
        """Returns the models data from <saved_model_dir>/models_data.json."""

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
