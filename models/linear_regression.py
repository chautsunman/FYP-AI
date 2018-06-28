from .model import Model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class LinearRegression(Model):
    def __init__(self):
        self.model = linear_model.LinearRegression()

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def error(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
