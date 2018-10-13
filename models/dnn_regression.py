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

    # Helper method to build the DNN model 
    def build_model(self):

        # Seed the machine
        np.random.seed()

        self.model = Sequential()

        # Specify the neural network configuration
        self.model.add(Dense(units=12, activation="relu", input_shape=(self.model_options["lookback"],)))
        self.model.add(Dense(units=8, activation="relu")) 
        self.model.add(Dense(units=1, activation="relu"))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    def __init__(self, model_options, load=False, saved_model_dir=None, saved_model_path=None):
        Model.__init__(self, model_options)

        if not load or saved_model_dir is None:
            self.build_model()

        else:
            model_path = saved_model_path if saved_model_path is not None else self.get_saved_model_path(saved_model_dir)
            if model_path is not None:
                self.model = load_model(saved_model_dir + "/" + model_path)

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
        
        print(x, y)
        loss = 500
        while loss >= 500:
            self.build_model()
            self.model.fit(x, y, epochs=200, batch_size=4)
            loss = self.model.evaluate(x, y)[1]
            print(loss)

    def predict(self, last_prices=None):
        if not self.model_options["use_stock_price"] and last_prices is None:
            return None

        x = last_prices.values[-self.model_options["lookback"]:].reshape(1, self.model_options["lookback"])

        results = []

        for i in range(1, 10):
            predictions = self.model.predict(x).flatten()
            results.append(predictions[0])

            # Use last record as new record's x
            x = np.roll(x, -1, axis=-1)
            x[0, -1] = predictions[0]

        if not self.model_options["use_stock_price"]:
            predictions[0] = last_prices[-1] * (1 + predictions[0])
            for i in range(1, predictions.shape[0]):
                predictions[i] = predictions[i - 1] * (1 + predictions[i])

        return np.array(results)

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
            json.dump(models_data, models_data_file)

    # Configuration options for a particular model
    def get_model_type(self):
        model_type = []
        model_type.append(str(self.model_options["n"]) + "days")
        model_type.append(str(self.model_options["lookback"]) + "lookback")
        model_type.append("change" if not self.model_options["use_stock_price"] else "price")
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
        options_name = [str(self.model_options["n"]), "days", "change" if not self.model_options["use_stock_price"] else "price", "lookback =", self.model_options["lookback"]]
        return "Dense Neural Network (%s)" % " ".join(options_name)

def get_all_predictions(stock_code, saved_model_dir, last_price):
    with open(saved_model_dir + "/models_data.json", "r") as models_data_file:
        models_data = json.load(models_data_file)

    if stock_code not in models_data["models"]:
        return [], []

    models_data = models_data["models"][stock_code]

    models = []
    for _, model_options in models_data.items():
        models.append(LinearRegression(model_options[-1], load=True, saved_model_dir=saved_model_dir, saved_model_path=model_options[-1]["model_path"]))

    predictions = []
    for model in models:
        if not model.model_options["use_stock_price"]:
            predictions.append(model.predict(last_price))
        else:
            predictions.append(model.predict())

    return predictions, models

