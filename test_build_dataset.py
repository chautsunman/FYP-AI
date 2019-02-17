from build_dataset import build_dataset
import pandas as pd
import numpy as np

input_options = {
    "config": [
        {"type": "lookback", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"},
        {"type": "moving_avg", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"}
    ],
    "stock_codes": ["GOOGL"],
    "stock_code": "GOOGL",
    "column": "adjusted_close"
}

predict_n = 1
training = False
stock_data = pd.read_csv('data/stock_prices/GOOGL.csv', index_col='timestamp').iloc[::-1]
stock_data = {
    'GOOGL': stock_data
}

mock_result = np.array([ 1100, 1120 ])

x = build_dataset(input_options, predict_n, training=training, stock_data=stock_data)
print("Before: ")
print(x)

x = build_dataset(input_options, predict_n, training=training, stock_data=stock_data, previous=mock_result)
print("After: ")
print(x)
