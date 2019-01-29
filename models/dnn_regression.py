import copy
from os import path
import pickle
import json
import time
import os

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, SimpleRNN, GRU
from keras import optimizers

import numpy as np
from sklearn.metrics import mean_squared_error

from models.model import Model

from build_dataset import get_input_shape
from options import OPTION_TYPES, rand_all, rand, mutate

class DenseNeuralNetwork(Model):
    """Neural network."""

    MODEL = "dnn"

    NETWORK_TYPES = ["dense", "SimpleRNN", "LSTM", "GRU"]

    MODEL_OPTIONS_CONFIG = {
        "net": {
            "type": OPTION_TYPES["nested"],
            "option_config": {
                "layers": {
                    "type": OPTION_TYPES["static"],
                    "value": [
                        {}
                    ],
                },
                "loss": {
                    "type": OPTION_TYPES["static"],
                    "value": "mse",
                    "option_config": {
                        "options": ["mse"]
                    }
                },
                "optimizer": {
                    "type": OPTION_TYPES["static"],
                    "value": "Adam",
                    "option_config": {
                        "options": ["sgd", "RMSprop", "Adam"]
                    }
                },
                "learning_rate": {
                    "type": OPTION_TYPES["discrete"],
                    "option_config": {
                        "options": [0.01, 0.001, 0.0001]
                    }
                },
                "epochs": {
                    "type": OPTION_TYPES["discrete"],
                    "option_config": {
                        "options": [10, 20, 50, 100]
                    }
                },
                "batch_size": {
                    "type": OPTION_TYPES["discrete"],
                    "option_config": {
                        "options": [16, 32, 64]
                    }
                },
                "metrics": {
                    "type": OPTION_TYPES["static"],
                    "value": ["mse"]
                }
            }
        },
        "predict_n": {
            "type": OPTION_TYPES["static"],
            "value": 10
        }
    }

    # options for each layer type
    LAYER_CONFIG = {
        "dense": {
            "layer_type": {
                "type": OPTION_TYPES["static"],
                "value": "dense"
            },
            "units": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": [8, 16, 32, 64, 128]
                }
            },
            "activation": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": ["relu", "tanh", "sigmoid", "linear"]
                }
            }
        },
        "SimpleRNN": {
            "layer_type": {
                "type": OPTION_TYPES["static"],
                "value": "SimpleRNN"
            },
            "units": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": [8, 16, 32, 64, 128]
                }
            },
            "activation": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": ["relu", "tanh", "sigmoid", "linear"]
                }
            },
            "return_sequences": {
                "type": OPTION_TYPES["static"],
                "value": True
            },
            "stateful": {
                "type": OPTION_TYPES["static"],
                "value": False
            }
        },
        "LSTM": {
            "layer_type": {
                "type": OPTION_TYPES["static"],
                "value": "LSTM"
            },
            "units": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": [8, 16, 32, 64, 128]
                }
            },
            "activation": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": ["relu", "tanh", "sigmoid", "linear"]
                }
            },
            "recurrent_activation": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": ["hard_sigmoid", "sigmoid"]
                }
            },
            "return_sequences": {
                "type": OPTION_TYPES["static"],
                "value": True
            },
            "stateful": {
                "type": OPTION_TYPES["static"],
                "value": False
            }
        },
        "GRU": {
            "layer_type": {
                "type": OPTION_TYPES["static"],
                "value": "GRU"
            },
            "units": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": [8, 16, 32, 64, 128]
                }
            },
            "activation": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": ["relu", "tanh", "sigmoid", "linear"]
                }
            },
            "recurrent_activation": {
                "type": OPTION_TYPES["discrete"],
                "option_config": {
                    "options": ["hard_sigmoid", "sigmoid"]
                }
            },
            "return_sequences": {
                "type": OPTION_TYPES["static"],
                "value": True
            },
            "stateful": {
                "type": OPTION_TYPES["static"],
                "value": False
            }
        }
    }

    OPTIMIZER_MAP = {
        "sgd": optimizers.SGD,
        "rmsprop": optimizers.RMSprop,
        "adagrad": optimizers.Adagrad,
        "adadelta": optimizers.Adadelta,
        "adam": optimizers.Adam,
        "adamax": optimizers.Adamax,
        "nadam": optimizers.Nadam
    }

    # initial layers for each network type
    INITIAL_LAYERS = {
        "dense": [
            {
                "layer_type": "dense"
            }
        ],
        "SimpleRNN": [
            {
                "layer_type": "SimpleRNN",
                "units": 10,
                "activation": "relu",
                "return_sequences": True,
                "stateful": False
            },
            {
                "layer_type": "dense"
            }
        ],
        "LSTM": [
            {
                "layer_type": "LSTM",
                "units": 10,
                "activation": "relu",
                "recurrent_activation": "hard_sigmoid",
                "return_sequences": True,
                "stateful": False
            },
            {
                "layer_type": "dense"
            }
        ],
        "GRU": [
            {
                "layer_type": "GRU",
                "units": 10,
                "activation": "relu",
                "recurrent_activation": "hard_sigmoid",
                "return_sequences": True,
                "stateful": False
            },
            {
                "layer_type": "dense"
            }
        ]
    }

    # mutations for each network type
    MUTATIONS = {
        "dense": [
            "add_dense_layer",
            "remove_dense_layer",
            "change_units",
            "change_activation",
            "learning_rate",
            "batch_size"
        ],
        "SimpleRNN": [
            "add_dense_layer",
            "remove_dense_layer",
            "add_rnn_layer",
            "remove_rnn_layer",
            "change_units",
            "change_activation",
            "learning_rate",
            "batch_size"
        ],
        "LSTM": [
            "add_dense_layer",
            "remove_dense_layer",
            "add_lstm_layer",
            "remove_lstm_layer",
            "change_units",
            "change_activation",
            "change_recurrent_activation",
            "learning_rate",
            "batch_size"
        ],
        "GRU": [
            "add_dense_layer",
            "remove_dense_layer",
            "add_gru_layer",
            "remove_gru_layer",
            "change_units",
            "change_activation",
            "change_recurrent_activation",
            "learning_rate",
            "batch_size"
        ]
    }

    def get_layer(self, layer_config, layer_type, is_input=False, is_output=False):
        """Return a layer based on layer_config and layer_type."""

        if layer_type == "SimpleRNN":
            if is_input and is_output:
                return SimpleRNN(
                    units=self.model_options["predict_n"],
                    activation=layer_config["activation"],
                    return_sequences=False,
                    stateful=layer_config["stateful"],
                    input_shape=self.input_shape
                )
            elif is_input:
                return SimpleRNN(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    return_sequences=layer_config["return_sequences"],
                    stateful=layer_config["stateful"],
                    input_shape=self.input_shape
                )
            elif is_output:
                return SimpleRNN(
                    units=self.model_options["predict_n"],
                    activation=layer_config["activation"],
                    return_sequences=False,
                    stateful=layer_config["stateful"]
                )
            else:
                return SimpleRNN(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    return_sequences=layer_config["return_sequences"],
                    stateful=layer_config["stateful"]
                )
        elif layer_type == "LSTM":
            if is_input and is_output:
                return LSTM(
                    units=self.model_options["predict_n"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=False,
                    stateful=layer_config["stateful"],
                    input_shape=self.input_shape
                )
            elif is_input:
                return LSTM(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=layer_config["return_sequences"],
                    stateful=layer_config["stateful"],
                    input_shape=self.input_shape
                )
            elif is_output:
                return LSTM(
                    units=layer_config["predict_n"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=False,
                    stateful=layer_config["stateful"]
                )
            else:
                return LSTM(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=layer_config["return_sequences"],
                    stateful=layer_config["stateful"]
                )
        elif layer_type == "GRU":
            if is_input and is_output:
                return GRU(
                    units=self.model_options["predict_n"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=False,
                    stateful=layer_config["stateful"],
                    input_shape=self.input_shape
                )
            elif is_input:
                return GRU(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=layer_config["return_sequences"],
                    stateful=layer_config["stateful"],
                    input_shape=self.input_shape
                )
            elif is_output:
                return GRU(
                    units=layer_config["predict_n"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=False,
                    stateful=layer_config["stateful"]
                )
            else:
                return GRU(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=layer_config["return_sequences"],
                    stateful=layer_config["stateful"]
                )
        else:
            if is_input and is_output:
                return Dense(
                    units=self.model_options["predict_n"],
                    input_shape=self.input_shape
                )
            elif is_input:
                return Dense(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    input_shape=self.input_shape
                )
            elif is_output:
                return Dense(
                    units=self.model_options["predict_n"]
                )
            else:
                return Dense(
                    units=layer_config["units"],
                    activation=layer_config["activation"]
                )

    # Helper method to build the DNN model
    def build_model(self):

        # Seed the machine
        np.random.seed()

        self.model = Sequential()

        net = self.model_options["net"]

        # Specify the neural network configuration
        if len(net["layers"]) == 1:
            self.model.add(self.get_layer(
                net["layers"][0],
                net["layers"][0]["layer_type"] if "layer_type" in net["layers"][0] else "dense",
                is_input=True,
                is_output=True
            ))
        else:
            self.model.add(self.get_layer(
                net["layers"][0],
                net["layers"][0]["layer_type"] if "layer_type" in net["layers"][0] else "dense",
                is_input=True
            ))
            for layer in net["layers"][1:-1]:
                self.model.add(self.get_layer(
                    layer,
                    layer["layer_type"] if "layer_type" in layer else "dense"
                ))
            self.model.add(self.get_layer(
                net["layers"][-1],
                net["layers"][-1]["layer_type"] if "layer_type" in net["layers"][-1] else "dense",
                is_output=True
            ))

        optimizer = self.OPTIMIZER_MAP[net["optimizer"].lower()](lr=net["learning_rate"])
        self.model.compile(loss=net["loss"], optimizer=optimizer, metrics=net["metrics"])

    def __init__(self, model_options, input_options, stock_code=None, load=False, saved_model_dir=None, saved_model_path=None):
        """Initializes the model. Creates a new model or loads a saved model."""

        Model.__init__(self, model_options, input_options, stock_code=stock_code)

        self.input_shape = get_input_shape(input_options)

        if not load or saved_model_dir is None:
            self.build_model()

        else:
            model_path = saved_model_path if saved_model_path is not None else self.get_saved_model_path(saved_model_dir)
            if model_path is not None:
                self.load_model(path.join(saved_model_dir, model_path), Model.KERAS_MODEL)

    def train(self, xs, ys, **kwargs):
        """Trains the model.

        Args:
            xs: A m-by-n NumPy data array of m data with n features.
            ys: A Numpy label array of m data.
        """

        self.model.fit(
            xs,
            ys,
            epochs=self.model_options["net"]["epochs"],
            batch_size=self.model_options["net"]["batch_size"],
            verbose=kwargs["verbose"] if "verbose" in kwargs else 1,
            callbacks=kwargs["callbacks"] if "callbacks" in kwargs else None
        )

        # # Initialize the evaluation_metric to its threshold so that the model must be trained
        # # at least once
        # evaluation_metric = self.model_options["net"]["evaluation_criteria"]["threshold"]

        # # If we aim to minimize the evaluation criteria, e.g. mse, retrain until criteria < threshold
        # if self.model_options["net"]["evaluation_criteria"]["minimize"]:
        #     while evaluation_metric >= self.model_options["net"]["evaluation_criteria"]["threshold"]:
        #         self.build_model()
        #         self.model.fit(xs, ys, epochs=self.model_options["net"]["epochs"], batch_size=self.model_options["net"]["batch_size"])
        #         evaluation_metric = self.model.evaluate(xs, ys)[1]
        # else:
        #     while evaluation_metric <= self.model_options["net"]["evaluation_criteria"]["threshold"]:
        #         self.build_model()
        #         self.model.fit(xs, ys, epochs=self.model_options["net"]["epochs"], batch_size=self.model_options["net"]["batch_size"])
        #         evaluation_metric = self.model.evaluate(xs, ys)[1]

    def predict(self, x):
        """Predicts.

        Returns:
            A NumPy array of the prediction.
        """

        predictions = self.model.predict(x)
        if x.shape[0] == 1:
            return predictions.flatten()
        return predictions

    # Save the models and update the models_data.json, which stores metadata of all DNN models
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

        self.create_model_dir(saved_model_dir)

        # Get the model name
        model_name = self.get_model_name()

        stock_code = "general" if self.stock_code is None else self.stock_code

        # Build the relative path of the model file
        model_path = path.join(self.get_model_type_hash(), stock_code)

        self.save_model(path.join(saved_model_dir, model_path, model_name), self.KERAS_MODEL)

        # Update the configuration file models_data.json, which stores metadata for all
        # the models built with DNN
        # Append to existing configuration file if there is one
        models_data = self.load_models_data(saved_model_dir)
        if models_data is None:
            # Create a new one if there is no configuration file for DNN yet
            models_data = {"models": {}, "modelTypes": {}}

        # update models data
        models_data = self.update_models_data(models_data, model_name, model_path)

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

        # model_type consists of all the parameters used for training this particular model
        # e.g. number of days used
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

    # Configuration options for a particular model
    def get_model_type(self):
        """Returns model type (model, model options, input options)."""

        return {"model": self.MODEL, "modelOptions": self.model_options, "inputOptions": self.input_options}

    def get_model_type_hash(self):
        """Returns model type hash."""

        model_type = self.get_model_type()

        model_type_json_str = self.get_json_str(model_type)

        return self.hash_str(model_type_json_str)

    # Build and get the model name
    # This implementation uses the model type plus a timestamp
    def get_model_name(self):
        """Returns model name (<model_type_hash>_<time>.model)."""

        model_name = []
        model_name.append(self.get_model_type_hash())
        model_name.append(str(int(time.time())))
        return "_".join(model_name) + ".h5"

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

    # Get the "Display name" for the model
    def get_model_display_name(self):
        """Returns model display name for the app."""

        if "network_type" in self.model_options:
            return "Neural Network, " + self.model_options["network_type"]
        else:
            return "Dense Neural Network"

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
                models.append(DenseNeuralNetwork(
                    models_data["modelTypes"][model_type]["modelOptions"],
                    models_data["modelTypes"][model_type]["inputOptions"],
                    stock_code=stock_code,
                    load=True,
                    saved_model_dir=saved_model_dir,
                    saved_model_path=path.join(
                        models_data["models"][model_type]["general"][-1]["model_path"],
                        models_data["models"][model_type]["general"][-1]["model_name"])
                ))
            if stock_code in models_data["models"][model_type] and len(models_data["models"][model_type][stock_code]) > 0:
                models.append(DenseNeuralNetwork(
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

    @staticmethod
    def random_models(n, network_type):
        models = []

        for _ in range(n):
            model_options = rand_all(DenseNeuralNetwork.MODEL_OPTIONS_CONFIG)
            model_options["network_type"] = network_type
            model_options["net"]["layers"] = DenseNeuralNetwork.INITIAL_LAYERS[network_type]

            input_options = {}
            if network_type == "dense":
                input_options = {
                    "config": [
                        {"type": "lookback", "n": 22, "stock_code": "GOOGL", "column": "adjusted_close"},
                        {"type": "moving_avg", "n": 5, "stock_code": "GOOGL", "column": "adjusted_close"},
                        {"type": "moving_avg", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"},
                        {"type": "moving_avg", "n": 30, "stock_code": "GOOGL", "column": "adjusted_close"},
                        {"type": "moving_avg", "n": 90, "stock_code": "GOOGL", "column": "adjusted_close"},
                        {"type": "moving_avg", "n": 180, "stock_code": "GOOGL", "column": "adjusted_close"},
                        {"type": "moving_avg", "n": 365, "stock_code": "GOOGL", "column": "adjusted_close"}
                    ],
                    "stock_codes": ["GOOGL"],
                    "stock_code": "GOOGL",
                    "column": "adjusted_close"
                }
            elif network_type in ["SimpleRNN", "LSTM", "GRU"]:
                input_options = {
                    "config": [
                        {"type": "lookback", "n": 1, "stock_code": "GOOGL", "column": "adjusted_close"}
                    ],
                    "stock_codes": ["GOOGL"],
                    "stock_code": "GOOGL",
                    "column": "adjusted_close",
                    "time_window": 10,
                    "time_window_offset": 1
                }

            models.append(DenseNeuralNetwork(
                model_options,
                input_options,
                "GOOGL"
            ))

        return models

    @staticmethod
    def evolve_model_options(parent_model_options, mutation):
        new_model_options = copy.deepcopy(parent_model_options)

        network_type = new_model_options["network_type"]

        parent_net = parent_model_options["net"]
        parent_layers = parent_net["layers"]

        if mutation == "add_dense_layer" and network_type == "dense":
            # add a dense layer to a dense network
            new_model_options["net"]["layers"].insert(
                np.random.randint(0, len(parent_layers)),
                rand_all(DenseNeuralNetwork.LAYER_CONFIG["dense"])
            )
        elif mutation == "add_dense_layer" and network_type in ["SimpleRNN", "LSTM", "GRU"]:
            # add a dense layer to a RNN, LSTM or GRU network
            first_dense_layer_idx = -1
            for layer_idx, layer in enumerate(parent_layers):
                if layer["layer_type"] == "dense":
                    first_dense_layer_idx = layer_idx
                    break
            new_model_options["net"]["layers"].insert(
                np.random.randint(first_dense_layer_idx, len(parent_layers)),
                rand_all(DenseNeuralNetwork.LAYER_CONFIG["dense"])
            )
        elif mutation == "add_rnn_layer" and network_type == "SimpleRNN":
            # add a RNN layer to a RNN network
            first_dense_layer_idx = -1
            for layer_idx, layer in enumerate(parent_layers):
                if layer["layer_type"] == "dense":
                    first_dense_layer_idx = layer_idx
                    break
            new_model_options["net"]["layers"].insert(
                np.random.randint(0, first_dense_layer_idx),
                rand_all(DenseNeuralNetwork.LAYER_CONFIG["SimpleRNN"])
            )
        elif mutation == "add_lstm_layer" and network_type == "LSTM":
            # add a LSTM layer to a LSTM network
            first_dense_layer_idx = -1
            for layer_idx, layer in enumerate(parent_layers):
                if layer["layer_type"] == "dense":
                    first_dense_layer_idx = layer_idx
                    break
            new_model_options["net"]["layers"].insert(
                np.random.randint(0, first_dense_layer_idx),
                rand_all(DenseNeuralNetwork.LAYER_CONFIG["LSTM"])
            )
        elif mutation == "add_gru_layer" and network_type == "GRU":
            # add a GRU layer to a GRU network
            first_dense_layer_idx = -1
            for layer_idx, layer in enumerate(parent_layers):
                if layer["layer_type"] == "dense":
                    first_dense_layer_idx = layer_idx
                    break
            new_model_options["net"]["layers"].insert(
                np.random.randint(0, first_dense_layer_idx),
                rand_all(DenseNeuralNetwork.LAYER_CONFIG["GRU"])
            )
        elif mutation == "remove_dense_layer" and network_type == "dense":
            # remove any dense layer except the output layer
            if len(parent_layers) > 1:
                dense_layer_idxs = []
                for layer_idx, layer in enumerate(parent_layers[:-1]):
                    if layer["layer_type"] == "dense":
                        dense_layer_idxs.append(layer_idx)
                new_model_options["net"]["layers"].pop(np.random.choice(dense_layer_idxs))
        elif mutation == "remove_dense_layer" and network_type in ["SimpleRNN", "LSTM", "GRU"]:
            # remove any dense layer from a RNN, LSTM or GRU network
            dense_layer_idxs = []
            for layer_idx, layer in enumerate(parent_layers[:-1]):
                if layer["layer_type"] == "dense":
                    dense_layer_idxs.append(layer_idx)
            if len(dense_layer_idxs) > 0:
                new_model_options["net"]["layers"].pop(np.random.choice(dense_layer_idxs))
        elif mutation == "remove_rnn_layer" and network_type == "SimpleRNN":
            # remove any RNN layer from a RNN network
            layer_idxs = []
            for layer_idx, layer in enumerate(parent_layers):
                if layer["layer_type"] == "SimpleRNN":
                    layer_idxs.append(layer_idx)
            if len(layer_idxs) > 1:
                new_model_options["net"]["layers"].pop(np.random.choice(layer_idxs))
        elif mutation == "remove_lstm_layer" and network_type == "LSTM":
            # remove any LSTM layer from a LSTM network
            layer_idxs = []
            for layer_idx, layer in enumerate(parent_layers):
                if layer["layer_type"] == "LSTM":
                    layer_idxs.append(layer_idx)
            if len(layer_idxs) > 1:
                new_model_options["net"]["layers"].pop(np.random.choice(layer_idxs))
        elif mutation == "remove_gru_layer" and network_type == "GRU":
            # remove any GRU layer from a GRU network
            layer_idxs = []
            for layer_idx, layer in enumerate(parent_layers):
                if layer["layer_type"] == "GRU":
                    layer_idxs.append(layer_idx)
            if len(layer_idxs) > 1:
                new_model_options["net"]["layers"].pop(np.random.choice(layer_idxs))
        elif mutation == "change_units":
            # change the number of units of a hidden layer
            if len(parent_layers) > 1:
                change_layer_idx = np.random.randint(0, len(parent_layers) - 1)
                change_layer_type = parent_layers[change_layer_idx]["layer_type"]
                new_model_options["net"]["layers"][change_layer_idx]["units"] = rand(
                    DenseNeuralNetwork.LAYER_CONFIG[change_layer_type]["units"]["type"],
                    DenseNeuralNetwork.LAYER_CONFIG[change_layer_type]["units"]["option_config"]
                )
        elif mutation == "change_activation":
            # change the activation of a hidden layer
            if len(parent_layers) > 1:
                change_layer_idx = np.random.randint(0, len(parent_layers) - 1)
                change_layer_type = parent_layers[change_layer_idx]["layer_type"]
                new_model_options["net"]["layers"][change_layer_idx]["activation"] = rand(
                    DenseNeuralNetwork.LAYER_CONFIG[change_layer_type]["activation"]["type"],
                    DenseNeuralNetwork.LAYER_CONFIG[change_layer_type]["activation"]["option_config"]
                )
        elif mutation == "change_recurrent_activation" and network_type in ["LSTM", "GRU"]:
            # change the recurrent activation of a layer
            layer_idxs = []
            for layer_idx, layer in enumerate(parent_layers):
                if layer["layer_type"] in ["LSTM", "GRU"]:
                    layer_idxs.append(layer_idx)
            change_layer_idx = np.random.choice(layer_idxs)
            change_layer_type = parent_layers[change_layer_idx]["layer_type"]
            new_model_options["net"]["layers"][change_layer_idx]["recurrent_activation"] = rand(
                DenseNeuralNetwork.LAYER_CONFIG[change_layer_type]["recurrent_activation"]["type"],
                DenseNeuralNetwork.LAYER_CONFIG[change_layer_type]["recurrent_activation"]["option_config"]
            )
        elif mutation in ["loss", "optimizer", "learning_rate", "epochs", "batch_size"]:
            # change the loss, optimizer, learning rate, number of epochs or batch size
            new_model_options["net"][mutation] = mutate(
                DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"][mutation]["type"],
                parent_net[mutation],
                DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"][mutation]["option_config"],
                1.0
            )

        return new_model_options

    @staticmethod
    def evolve(models, n):
        """Cross-over and breed new models."""

        new_models = models
        mutations = []

        best_model_options = [model.model_options for model in models]

        # create n - len(models) models
        while len(new_models) < n:
            # choose a parent model
            parent_model_idx = np.random.randint(len(best_model_options))
            parent_model = models[parent_model_idx]

            # randomly choose a mutation
            mutation = np.random.choice(DenseNeuralNetwork.MUTATIONS[parent_model.model_options["network_type"]])
            mutations.append(mutation)

            # mutate the model
            new_model_options = DenseNeuralNetwork.evolve_model_options(parent_model.model_options, mutation)

            # create a new model
            new_models.append(DenseNeuralNetwork(
                new_model_options,
                parent_model.input_options,
                parent_model.stock_code
            ))

        return new_models, mutations
