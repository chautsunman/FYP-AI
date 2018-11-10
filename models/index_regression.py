from models.model import Model

class IndexRegressionModel(Model):
    def __init__(self, model_options, stock_code):
        Model.__init__(self, model_options, stock_code=stock_code)

    def train(self, stock_prices):
        pass

    def predict(self):
        return None

    def save(self, saved_model_dir):
        pass

    def get_model_display_name(self):
        return "Index Regression Model"

    def error(self, y_true, y_pred):
        return 0
