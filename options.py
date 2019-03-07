import math

import numpy as np

OPTION_TYPES = {
    "discrete": "discrete",
    "range": "range",
    "continuous": "continuous",
    "step": "step",
    "nested": "nested",
    "array": "array",
    "static": "static"
}

class DiscreteOption(object):
    #option_config = {options:[1,2,3,4]}

    TYPE = OPTION_TYPES["discrete"]

    @staticmethod
    def rand(option_config):
        return option_config["options"][np.random.randint(len(option_config["options"]))]

    @staticmethod
    def mutate(option, option_config):
        # print (option_config)
        return DiscreteOption.rand(option_config)

class RangeOption(object):
    TYPE = OPTION_TYPES["range"]

    @staticmethod
    def rand(option_config):
        return np.random.randint(option_config["range"][0], option_config["range"][1] + 1)

    @staticmethod
    def mutate(option, option_config):
        low = max(math.floor(option / 2), option_config["range"][0])
        high = min(math.ceil(option * 2), option_config["range"][1])
        return np.random.randint(low, high + 1)

class ContinuousOption(object):
    #option_config = {range:[1,10]}

    TYPE = OPTION_TYPES["continuous"]

    @staticmethod
    def rand(option_config):
        return np.random.rand() * (option_config["range"][1] - option_config["range"][0]) + option_config["range"][0]

    @staticmethod
    def mutate(option, option_config):
        ranNum = np.random.randn()
        if (option + ranNum) > option_config["range"][1]:
            return option_config["range"][1]
        elif (option + ranNum) < option_config["range"][0]:
            return option_config["range"][0]
        else:
            return option + ranNum

class StepOption(object):
    #option_config = {range:[1,10, STEP]}

    TYPE = OPTION_TYPES["step"]

    @staticmethod
    def rand(option_config):
        return np.random.choice(np.arange(
            option_config["range"][0],
            option_config["range"][1] + option_config["step"],
            option_config["step"]
        ))

    @staticmethod
    def mutate(option, option_config):
        ranNum = np.random.choice([option_config["step"], -option_config["step"]])
        if (option + ranNum) > option_config["range"][1]:
            return option_config["range"][1]
        elif (option + ranNum) < option_config["range"][0]:
            return option_config["range"][0]
        else:
            return option + ranNum

def rand(option_type, option_config):
    if option_type == DiscreteOption.TYPE:
        return DiscreteOption.rand(option_config)
    elif option_type == RangeOption.TYPE:
        return RangeOption.rand(option_config)
    elif option_type == ContinuousOption.TYPE:
        return ContinuousOption.rand(option_config)
    elif option_type == StepOption.TYPE:
        return StepOption.rand(option_config)


def mutate(option_type, option, option_config, probability=0.2):
    if np.random.rand() < probability:
        if option_type == DiscreteOption.TYPE:
            return DiscreteOption.mutate(option, option_config)
        elif option_type == RangeOption.TYPE:
            return RangeOption.mutate(option, option_config)
        elif option_type == ContinuousOption.TYPE:
            return ContinuousOption.mutate(option, option_config)
        elif option_type == StepOption.TYPE:
            return StepOption.mutate(option, option_config)
    else:
        return option

def rand_all(configs):
    results = {}
    for key in configs:
        if configs[key]["type"] == OPTION_TYPES["nested"]:
            results[key] = rand_all(configs[key]["option_config"])
        elif configs[key]["type"] == OPTION_TYPES["array"]:
            results[key] = [rand(
                option_config["type"],
                option_config["option_config"]) for option_config in configs[key]["option_configs"]]
        elif configs[key]["type"] == OPTION_TYPES["static"]:
            results[key] = configs[key]["value"]
        else:
            results[key] = rand(configs[key]["type"], configs[key]["option_config"])
    return results

def cross_over_all(config, options):
    """Cross-over options."""

    new_options = {}

    for option in config:
        if config[option]["type"] == OPTION_TYPES["nested"]:
            new_options[option] = cross_over_all(
                config[option]["option_config"],
                [o[option] for o in options]
            )
        elif config[option]["type"] == OPTION_TYPES["array"]:
            new_options[option] = [
                cross_over_all(
                    option_config,
                    [o[option][i] for o in options]
                )
                for i, option_config in enumerate(config[option]["option_configs"])
            ]
        elif config[option]["type"] == OPTION_TYPES["static"]:
            new_options[option] = config[option]["value"]
        else:
            new_options[option] = options[np.random.randint(len(options))][option]

    return new_options

def mutate_all(options, configs, probability=0.2):
    results = {}
    for key in configs:
        if configs[key]["type"] == OPTION_TYPES["nested"]:
            results[key] = mutate_all(options[key],configs[key]["option_config"], probability)
        elif configs[key]["type"] == OPTION_TYPES["array"]:
            results[key] = [mutate(
                option_config["type"],
                options[key][i],
                option_config["option_config"],
                probability) for i, option_config in enumerate(configs[key]["option_configs"])]
        elif configs[key]["type"] == OPTION_TYPES["static"]:
            results[key] = configs[key]["value"]
        else:
            results[key] = mutate(configs[key]["type"],options[key],configs[key]["option_config"], probability)
    return results
