import argparse
import json

from train_models import train_models 

from models.linear_index_regression import LinearIndexRegression
from models.svr_index_regression import SupportVectorIndexRegression

def index_model_scoring(model_data):
    if model_data["model"] == "linear_regression":
        print ("Squared Error", LinearIndexRegression.calculate_average_mean_squared_error(model_data["modelOptions"], "./data"))
    elif model_data["model"] == "svr":
        print ("Squared Error SVM Index", SupportVectorIndexRegression.calculate_average_mean_squared_error(model_data["modelOptions"], "./data"))
    else:
        print ("Failed to locate model data. ")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models.")
    parser.add_argument("train_models_json", help="Train models JSON file path")

    args = parser.parse_args()

    with open("./" + args.train_models_json) as train_models_json_file:
        train_models_data = json.load(train_models_json_file)

    index_model_scoring(train_models_data)