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
                self.model = load_model(saved_model_dir + "/" + model_path)

    # Build input, ouput for training
    # Transform time-series data to a dataset modeling stock price at t
    # given stock prices at previous time periods
    # i.e. ((p0, p1, p2...), pn)
    def build_lookback(self, data, lookback=1):
        return np.stack(data[i:i+lookback] for i in range(0, data.shape[0]-lookback)), data[lookback:]

    def train(self, stock_prices):
        if not self.model_options["use_stock_price"]:
            data = stock_prices["change"]
        else:
            data = stock_prices["adjusted_close"]

        # Reverse order of the data
        data = np.flipud(data.values.reshape(-1))
        x, y = self.build_lookback(data, self.model_options["lookback"])

        # Initialize the evaluation_metric to its threshold so that the model must be trained
        # at least once
        evaluation_metric = self.model_options["net"]["evaluation_criteria"]["threshold"]

        # If we aim to minimize the evaluation criteria, e.g. mse, retrain until criteria < threshold
        if self.model_options["net"]["evaluation_criteria"]["minimize"]:
            while evaluation_metric >= self.model_options["net"]["evaluation_criteria"]["threshold"]:
                self.build_model()
                self.model.fit(x, y, epochs=self.model_options["net"]["epochs"], batch_size=self.model_options["net"]["batch_size"])
                evaluation_metric = self.model.evaluate(x, y)[1]
        else:
            while evaluation_metric <= self.model_options["net"]["evaluation_criteria"]["threshold"]:
                self.build_model()
                self.model.fit(x, y, epochs=self.model_options["net"]["epochs"], batch_size=self.model_options["net"]["batch_size"])
                evaluation_metric = self.model.evaluate(x, y)[1]

    def predict(self, lookback_data, last_price=None):
        if not self.model_options["use_stock_price"] and last_price is None:
            return None

        if not self.model_options["use_stock_price"]:
            data = lookback_data["change"]
        else:
            data = lookback_data["adjusted_close"]

        x = np.flipud(data.values.reshape(1, -1))
        print(x)

        predictions = []

        for i in range(0, 30):
            results = self.model.predict(x).flatten()
            predictions.append(results[0])

            # Use last record as new record's x
            x = np.roll(x, -1, axis=-1)
            x[0, -1] = results[0]

        print(predictions)

        if not self.model_options["use_stock_price"]:
            predictions[0] = last_price * (1 + predictions[0])
            for i in range(1, len(predictions)):
                predictions[i] = predictions[i - 1] * (1 + predictions[i])

        return np.array(predictions)

    # Save the models and update the models_data.json, which stores metadata of all DNN models
    def save(self, saved_model_dir):
        Model.create_model_dir(self, saved_model_dir + "/" + self.model_options["stock_code"])

        # Get the model name
        model_name = self.get_model_name()

        # Build the relative path of the model file
        model_path = self.model_options["stock_code"] + "/" + model_name

        fullpath = saved_model_dir + "/" + model_path
        self.model.save(fullpath)

        # Update the configuration file models_data.json, which stores metadata for all
        # the models built with DNN
        if os.path.isfile(saved_model_dir + "/" + "models_data.json"):
            # Append to existing configuration file if there is one
            with open(saved_model_dir + "/" + "models_data.json", "r") as models_data_file:
                models_data = json.load(models_data_file)
        else:
            # Create a new one if there is no configuration file for DNN yet
            models_data = {"models": {}}

        if self.model_options["stock_code"] not in models_data["models"]:
            models_data["models"][self.model_options["stock_code"]] = {}

        # model_type consists of all the parameters used for training this particular model
        # e.g. number of days used
        model_type = self.get_model_type()

        if model_type not in models_data["models"][self.model_options["stock_code"]]:
            models_data["models"][self.model_options["stock_code"]][model_type] = []

        model_data = self.model_options
        model_data["model_name"] = model_name
        model_data["model_path"] = model_path

        models_data["models"][self.model_options["stock_code"]][model_type].append(model_data)

        with open(saved_model_dir + "/" + "models_data.json", "w") as models_data_file:
            json.dump(models_data, models_data_file, indent=4)

    # Configuration options for a particular model
    def get_model_type(self):
        model_type = []
        model_type.append(str(self.model_options["n"]) + "days")
        model_type.append(str(self.model_options["lookback"]) + "lookback")
        model_type.append("change" if not self.model_options["use_stock_price"] else "price")
        model_type.append(self.model_options["net"]["name"])
        return "_".join(model_type)

    # Build and get the model name
    # This implementation uses the model type plus a timestamp
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

    # Get the "Display name" for the model
    def get_model_display_name(self):
        options_name = [str(self.model_options["n"]), "days", "change" if not self.model_options["use_stock_price"] else "price", "lookback =", str(self.model_options["lookback"])]
        return "Dense Neural Network (%s)" % " ".join(options_name)

def get_all_predictions(stock_code, saved_model_dir, lookback_data, last_price):
    with open(saved_model_dir + "/models_data.json", "r") as models_data_file:
        models_data = json.load(models_data_file)

    if stock_code not in models_data["models"]:
        return [], []

    models_data = models_data["models"][stock_code]

    models = []
    for _, model_options in models_data.items():
        models.append(DenseNeuralNetwork(model_options[-1], load=True, saved_model_dir=saved_model_dir, saved_model_path=model_options[-1]["model_path"]))

    predictions = []
    for model in models:
        if not model.model_options["use_stock_price"]:
            predictions.append(model.predict(lookback_data, last_price))
        else:
            predictions.append(model.predict(lookback_data))

    return predictions, models

def get_no_of_data_required(stock_code, saved_model_dir):
    with open(saved_model_dir + "/models_data.json", "r") as models_data_file:
        models_data = json.load(models_data_file)

    if stock_code not in models_data["models"]:
        return 0

    models_data = models_data["models"][stock_code]

    no_required = 0
    for _, model_options in models_data.items():
        if model_options[-1]["lookback"] > no_required:
            no_required = model_options[-1]["lookback"]

    return no_required

