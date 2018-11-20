import argparse
import json

from train_models import train_models 

from models.linear_index_regression import LinearIndexRegression
from models.svr_index_regression import SupportVectorIndexRegression
import matplotlib.pyplot as plt

def index_model_scoring(model_data):
    
    if model_data["model"] == LinearIndexRegression.MODEL:
        error = LinearIndexRegression.calculate_average_mean_squared_error(
            model_data["modelOptions"],
            model_data["inputOptions"],
            model_data["inputOptions"]["stock_code"],
            3,
            "./data")
    elif model_data["model"] == SupportVectorIndexRegression.MODEL:
        error = SupportVectorIndexRegression.calculate_average_mean_squared_error(
            model_data["modelOptions"],
            model_data["inputOptions"],
            model_data["inputOptions"]["stock_code"],
            3,
            "./data")
    else:
        error = 0
    return error

def stress_test_models(train_models_data):
    # for linear_index_regression
    # modify the n in the model_data then call the index_model_scoring to get the error
    for model_data in train_models_data["models"]:
        x = []
        y = []
        
        #scenario for changing n, num of days used for regression
        if model_data["model"] == LinearIndexRegression.MODEL:
            for i in range(2,20,1): #loop from 2 to 10
                model_data["inputOptions"]["config"][0]["n"] = i #set the parameter you want to change for doing the stress_test
                error = index_model_scoring(model_data)
                x.append(i)
                y.append(error)
                # print("when n is ", i, " error is ", error)
            plt.plot(x,y, label='linear_index_regression')
            plt.xlabel('n (num of days used)')
            plt.ylabel('average mean square error')
            plt.title('Stress Test Models (changing n)')
        elif model_data["model"] == SupportVectorIndexRegression.MODEL:
            for i in range(2,20,1): #loop from 2 to 10
                model_data["inputOptions"]["config"][0]["n"] = i #set the parameter you want to change for doing the stress_test
                error = index_model_scoring(model_data)
                x.append(i)
                y.append(error)
                # print("when n is ", i, " error is ", error)
            plt.plot(x,y, label='svr_index_regression')
            plt.xlabel('n (num of days used)')
            plt.ylabel('average mean square error')
            plt.title('Stress Test Models (changing n)')
        
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models.")
    parser.add_argument("train_models_json", help="Train models JSON file path")

    args = parser.parse_args()

    with open("./" + args.train_models_json) as train_models_json_file:
        train_models_data = json.load(train_models_json_file)

    for model_data in train_models_data["models"]:
        if model_data["model"] in [LinearIndexRegression.MODEL, SupportVectorIndexRegression.MODEL]:
            error = index_model_scoring(model_data)
    
    #stress test model
    stress_test_models(train_models_data)
