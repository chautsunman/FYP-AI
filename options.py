import numpy as np

class DiscreteOption(Object):
    #option_config = {options:[1,2,3,4]}

    TYPE = "discrete"

    @staticmethod
    def rand(option_config):
        return np.random.choice(option_config["options"])

    @staticmethod
    def mutate(option, option_config):
        return DiscreteOption.rand(option_config)
    
class ContinuousOption(Object):
    #option_config = {range:[1,10]}

    TYPE = "continuous"

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

def rand(option_type, option_config):
    if option_type == DiscreteOption.TYPE:
        return DiscreteOption.rand(option_config)
    elif option_type == ContinuousOption.TYPE:
        return ContinuousOption.rand(option_config)

def mutate(option_type, option, option_config, probability=0.2):
    if np.random.rand() < probability: 
        if option_type == DiscreteOption.TYPE:
            return DiscreteOption.mutate(option, option_config)
        elif option_type == ContinuousOption.TYPE:
            return ContinuousOption.mutate(option, option_config)

def rand_all(configs):
    results = {}
    for key in configs:
        results[key] = rand(configs[key]["type"],configs[key]["option_config"])
    return results

def mutate_all(options, configs, probability=0.2):
    results = {}
    for key in configs:
        results[key] = mutate(configs[key]["type"],options[key],configs[key]["option_config"], probability)
    return results
    