from os import path
import pickle
import json
import time
import os

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import optimizers

import numpy as np
from sklearn.metrics import mean_squared_error

from models.model import Model

class DenseNeuralNetwork(Model):
    MODEL = "dnn"

    # Helper method to build the DNN model
    def build_model(self):

        # Seed the machine
        np.random.seed()

        self.model = Sequential()

        net = self.model_options["net"]

        # Specify the neural network configuration
        for layer in net["layers"]:
            if "is_input" in layer and layer["is_input"]:
                self.model.add(Dense(units=layer["units"], activation=layer["activation"], input_shape=(self.model_options["lookback"],)))
            elif "is_output" in layer and layer["is_output"]:
                self.model.add(Dense(units=1, activation=layer["activation"]))
            else:
                self.model.add(Dense(units=layer["units"], activation=layer["activation"]))
        #self.model.add(Dense(units=12, activation="relu", input_shape=(self.model_options["lookback"],)))
        #self.model.add(Dense(units=8, activation="relu"))
        #self.model.add(Dense(units=1, activation="relu"))

        #self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        self.model.compile(loss=net["loss"], optimizer=net["optimizer"], metrics=net["metrics"])

    def __init__(self, model_options, load=False, saved_model_dir=None, saved_model_path=None):
        Model.__init__(self, model_options)

        if not load or saved_model_dir is None:
            self.build_model()

        else:
            model_path = saved_model_path if saved_model_path is not None else self.get_saved_model_path(saved_model_dir)
            if model_path is not None:
                self.load_model(path.join(saved_model_dir, model_path))

    def train(self, xs, ys):
        # Initialize the evaluation_metric to its threshold so that the model must be trained
        # at least once
        evaluation_metric = self.model_options["net"]["evaluation_criteria"]["threshold"]

        # If we aim to minimize the evaluation criteria, e.g. mse, retrain until criteria < threshold
        if self.model_options["net"]["evaluation_criteria"]["minimize"]:
            while evaluation_metric >= self.model_options["net"]["evaluation_criteria"]["threshold"]:
                self.build_model()
                self.model.fit(xs, ys, epochs=self.model_options["net"]["epochs"], batch_size=self.model_options["net"]["batch_size"])
                evaluation_metric = self.model.evaluate(xs, ys)[1]
        else:
            while evaluation_metric <= self.model_options["net"]["evaluation_criteria"]["threshold"]:
                self.build_model()
                self.model.fit(xs, ys, epochs=self.model_options["net"]["epochs"], batch_size=self.model_options["net"]["batch_size"])
                evaluation_metric = self.model.evaluate(xs, ys)[1]

    def predict(self, x):
        return self.model.predict(x)

    # Save the models and update the models_data.json, which stores metadata of all DNN models
    def save(self, saved_model_dir):
        self.create_model_dir(self, path.join(saved_model_dir, self.model_options["stock_code"]))

        # Get the model name
        model_name = self.get_model_name()

        # Build the relative path of the model file
        model_path = path.join(self.model_options["stock_code"], model_name)

        self.save_model(path.join(saved_model_dir, model_path), self.KERAS_MODEL)

        # Update the configuration file models_data.json, which stores metadata for all
        # the models built with DNN
        # Append to existing configuration file if there is one
        models_data = self.load_models_data(saved_model_dir)
        if models_data is None:
            # Create a new one if there is no configuration file for DNN yet
            models_data = {"models": [], "modelTypes": {}}

        # update models data
        models_data = self.update_models_data(models_data, model_name, model_path)

        self.save_models_data(models_data, saved_model_dir)

    def update_models_data(self, models_data, model_name, model_path):
        # model_type consists of all the parameters used for training this particular model
        # e.g. number of days used
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

    # Configuration options for a particular model
    def get_model_type(self):
        return {"model": self.MODEL, "modelOptions": self.model_options}

    def get_model_type_hash(self):
        model_type = self.get_model_type()

        model_type_json_str = self.get_json_str(model_type)

        return self.hash_str(model_type_json_str)

    # Build and get the model name
    # This implementation uses the model type plus a timestamp
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

    # Get the "Display name" for the model
    def get_model_display_name(self):
        options_name = [str(self.model_options["n"]), "days", "change" if not self.model_options["use_stock_price"] else "price", "lookback =", str(self.model_options["lookback"])]
        return "Dense Neural Network (%s)" % " ".join(options_name)

    @staticmethod
    def get_all_predictions(stock_code, saved_model_dir):
        models_data = Model.load_models_data(saved_model_dir)
        if models_data is None:
            return None

        models = []
        for model_type, model_data in models_data["models"].items():
            models.append(DenseNeuralRegression(
                models_data["modelTypes"][model_type]["modelOptions"],
                load=True,
                saved_model_dir=saved_model_dir,
                saved_model_path=model_data[-1]["model_path"]))

        predictions = []
        for model in models:
            predictions.append(np.array([]))

        return predictions, models
