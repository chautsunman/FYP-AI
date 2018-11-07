import numpy as np

class Option(Object):
    #base class
    def __init__(self):
        pass
    
    def rand(self):
        pass

    def mutate(self, option):
        pass

class DiscreteOption(Option):
    def __init__(self, options):
        self.options = options

    def rand(self):
        return np.random.choice(self.options)

    def mutate(self, option):
        return self.rand()
    
class ContinuousOption(Option):
    def __init__(self, range_options):
        self.range_options = range_options

    def rand(self):
        return np.random.rand() * (self.range_options[1] - self.range_options[0]) + self.range_options[0]

    def mutate(self, option):
        return option + np.random.randn()
