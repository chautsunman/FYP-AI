from models.model import Model

class IndexRegressionModel(Model):
    """Base class for all index regression models."""

    def __init__(self, model_options, input_options, stock_code):
        """Initializes the model."""

        Model.__init__(self, model_options, input_options, stock_code=stock_code)

    def train(self, xs, ys):
        """Trains the model."""
        pass

    def predict(self):
        """Predicts."""
        return None

    def save(self, saved_model_dir):
        """Saves the model in saved_model_dir."""
        pass

    def get_model_display_name(self):
        return "Index Regression Model"

    def error(self, y_true, y_pred):
        return 0
