import argparse
import json

from train_models import train_models 

from models.linear_index_regression import LinearIndexRegression
from models.svr_index_regression import SupportVectorIndexRegression

def index_model_scoring(model_data):
    
    if model_data["model"] == LinearIndexRegression.MODEL:
        error = LinearIndexRegression.calculate_average_mean_squared_error(
            model_data["modelOptions"],
            model_data["inputOptions"],
            model_data["inputOptions"]["stock_code"],
            2,
            "./data")
        print ("Squared Error", error)
    elif model_data["model"] == SupportVectorIndexRegression.MODEL:
        error = SupportVectorIndexRegression.calculate_average_mean_squared_error(
            model_data["modelOptions"],
            model_data["inputOptions"],
            model_data["inputOptions"]["stock_code"],
            2,
            "./data")
        print ("Squared Error SVM Index", error)
    else:
        error = 0
        print ("Failed to locate model data. ")
    
    return error
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models.")
    parser.add_argument("train_models_json", help="Train models JSON file path")

    args = parser.parse_args()

    with open("./" + args.train_models_json) as train_models_json_file:
        train_models_data = json.load(train_models_json_file)

    for model_data in train_models_data["models"]:
        if model_data["model"] in [LinearIndexRegression.MODEL, SupportVectorIndexRegression.MODEL]:
            error = index_model_scoring(model_data)
