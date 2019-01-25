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

    MODEL_OPTIONS_CONFIG = {
        "net": {
            "type": OPTION_TYPES["nested"],
            "option_config": {
                "layers": {
                    "type": OPTION_TYPES["static"],
                    "value": [
                        {}
                    ],
                    "layer": {
                        "units": {
                            "type": OPTION_TYPES["discrete"],
                            "option_config": {
                                "options": [4, 8, 16, 32, 64, 128]
                            }
                        },
                        "activation": {
                            "type": OPTION_TYPES["discrete"],
                            "option_config": {
                                "options": ["relu", "tanh", "sigmoid", "linear"]
                            }
                        }
                    },
                    "option_configs": [
                        {
                            "units": {
                                "type": OPTION_TYPES["step"],
                                "option_config": {
                                    "range": [1, 10],
                                    "step": 1
                                }
                            },
                            "activation": {
                                "type": OPTION_TYPES["discrete"],
                                "option_config": {
                                    "options": [
                                        "relu", "linear", "exponential", "hard_sigmoid", "sigmoid",
                                        "tanh", "softsign", "softplus", "selu", "elu", "softmax"
                                    ]
                                }
                            },
                            "recurrent_activation": {
                                "type": OPTION_TYPES["discrete"],
                                "option_config": {
                                    "options": ["hard_sigmoid", "sigmoid"]
                                }
                            },
                            "stateful": {
                                "type": OPTION_TYPES["discrete"],
                                "option_config": {
                                    "options": [True, False]
                                }
                            }
                            # "is_input": {
                            #     "type": OPTION_TYPES["discrete"],
                            #     "option_config": {
                            #         "options": [True, False]
                            #     }
                            # },
                            # "inputUnits": {
                            #     "type": OPTION_TYPES["step"],
                            #     "option_config": {
                            #         "range": [10, 20],
                            #         "step": 5
                            #     }
                            # }
                        }
                    ]
                },
                "loss": {
                    "type": OPTION_TYPES["discrete"],
                    "option_config": {
                        "options": ["mse"]
                    }
                },
                "optimizer": {
                    "type": OPTION_TYPES["discrete"],
                    "option_config": {
                        "options": ["sgd", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
                    }
                },
                "learning_rate": {
                    "type": OPTION_TYPES["discrete"],
                    "option_config": {
                        "options": [0.1, 0.01, 0.001, 0.0001]
                    }
                },
                "epochs": {
                    "type": OPTION_TYPES["step"],
                    "option_config": {
                        "range": [1, 10],
                        "step": 1
                    }
                },
                "batch_size": {
                    "type": OPTION_TYPES["discrete"],
                    "option_config": {
                        "options": [1, 8, 16, 32, 64]
                    }
                },
                "metrics": {
                    "type": OPTION_TYPES["static"],
                    "value": ["mse"]
                },
                "evaluation_criteria": {
                    "type": OPTION_TYPES["nested"],
                    "option_config": {
                        "minimize": {
                            "type": OPTION_TYPES["static"],
                            "value": False
                        },
                        "threshold": {
                            "type": OPTION_TYPES["continuous"],
                            "option_config": {
                                "range": [1, 10]
                            }
                        }
                    }
                }
            }
        },
        "predict_n": {
            "type": OPTION_TYPES["static"],
            "value": 10
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
                    recurrent_activation=layer_config["recurrent_activation"],
                    return_sequences=False,
                    stateful=layer_config["stateful"]
                )
            else:
                return SimpleRNN(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    recurrent_activation=layer_config["recurrent_activation"],
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
                net["layers"][0]["type"] if "type" in net["layers"] else "dense",
                is_input=True,
                is_output=True
            ))
        else:
            self.model.add(self.get_layer(
                net["layers"][0],
                net["layers"][0]["type"] if "type" in net["layers"] else "dense",
                is_input=True
            ))
            for layer in net["layers"][1:-1]:
                self.model.add(self.get_layer(
                    layer,
                    layer["type"] if "type" in net["layers"] else "dense"
                ))
            self.model.add(self.get_layer(
                net["layers"][-1],
                net["layers"][-1]["type"] if "type" in net["layers"] else "dense",
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

    def train(self, xs, ys):
        """Trains the model.

        Args:
            xs: A m-by-n NumPy data array of m data with n features.
            ys: A Numpy label array of m data.
        """

        self.model.fit(xs, ys, epochs=self.model_options["net"]["epochs"], batch_size=self.model_options["net"]["batch_size"])

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
    def random_models(n):
        return [
            DenseNeuralNetwork(
                rand_all(DenseNeuralNetwork.MODEL_OPTIONS_CONFIG),
                {
                    "config": [
                        {"type": "lookback", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"},
                        {"type": "moving_avg", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"}
                    ],
                    "stock_codes": ["GOOGL"],
                    "stock_code": "GOOGL",
                    "column": "adjusted_close"
                },
                "GOOGL"
            )
            for _ in range(n)
        ]

    @staticmethod
    def evolve(models, n):
        """Cross-over and breed new models."""

        new_models = models

        best_model_options = [model.model_options for model in models]

        while len(new_models) < n:
            new_model_options = np.random.choice(best_model_options)

            change_option = np.random.choice([
                "add_layer",
                "remove_layer",
                "change_units",
                "change_activation",
                "loss",
                "optimizer",
                "learning_rate",
                "epochs",
                "batch_size"
            ])

            if change_option == "add_layer":
                new_model_options["net"]["layers"].insert(
                    np.random.randint(0, len(new_model_options["net"]["layers"])),
                    rand_all(DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"]["layers"]["layer"])
                )
            elif change_option == "remove_layer":
                if len(new_model_options["net"]["layers"]) > 1:
                    new_model_options["net"]["layers"].pop(np.random.randint(0, len(new_model_options["net"]["layers"]) - 1))
            elif change_option == "change_units":
                if len(new_model_options["net"]["layers"]) > 1:
                    change_layer_idx = np.random.randint(0, len(new_model_options["net"]["layers"]) - 1)
                    new_model_options["net"]["layers"][change_layer_idx]["units"] = rand(
                        DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"]["layers"]["layer"]["units"]["type"],
                        DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"]["layers"]["layer"]["units"]["option_config"]
                    )
            elif change_option == "change_activation":
                if len(new_model_options["net"]["layers"]) > 1:
                    change_layer_idx = np.random.randint(0, len(new_model_options["net"]["layers"]) - 1)
                    new_model_options["net"]["layers"][change_layer_idx]["activation"] = rand(
                        DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"]["layers"]["layer"]["activation"]["type"],
                        DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"]["layers"]["layer"]["activation"]["option_config"]
                    )
            elif change_option in ["loss", "optimizer", "learning_rate", "epochs", "batch_size"]:
                new_model_options["net"][change_option] = mutate(
                    DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"][change_option]["type"],
                    new_model_options["net"][change_option],
                    DenseNeuralNetwork.MODEL_OPTIONS_CONFIG["net"]["option_config"][change_option]["option_config"],
                    1.0
                )

            new_models.append(DenseNeuralNetwork(
                new_model_options,
                {
                    "config": [
                        {"type": "lookback", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"},
                        {"type": "moving_avg", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"}
                    ],
                    "stock_codes": ["GOOGL"],
                    "stock_code": "GOOGL",
                    "column": "adjusted_close"
                },
                "GOOGL"
            ))

        return new_models
