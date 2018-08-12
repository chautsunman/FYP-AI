import os

class Model:
    def __init__(self, model_options):
        self.model = None
        self.model_options = model_options

    def train(self):
        return

    def predict(self):
        return None

    def save(self, saved_model_dir):
        return

    def create_model_dir(self, model_dir_path):
        if not os.path.isdir(model_dir_path):
            os.makedirs(model_dir_path)
