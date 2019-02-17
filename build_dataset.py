import numpy as np
import pandas as pd

def get_sliding_window(data, window_size):
    """Get a sliding window.
    
    Args:
        data: an n-dimensional ndarray of shape (d1, d2, ..., dn)
        window_size: the window size
        slide: how many elements to per window
    
    Returns:
        An (n+1) dimensional ndarray of shape (d1, d2, ..., dn, dn+1)
        where a subarray of shape (d2, ..., dn+1) is a window
    """
    strides = (data.strides[0],) + data.strides
    dataset_size = data.shape[0] - window_size + 1

    
    if len(data.shape) > 1:
        shape = (dataset_size, window_size) + data.shape[1:]
    else:
        shape = (dataset_size, window_size)

    print("Shape: {}, Stride: {}".format(shape, strides))

    #data_start = data.shape[0]
    #return np.lib.stride_tricks.as_strided(data[data_start:], shape, strides)
    return np.lib.stride_tricks.as_strided(data[:], shape, strides)

def get_moving_avg(stock_data, stock_code, column, n, skip_last = 0, **kwargs):
    """Get moving avg
    Args:
        stock_data: pandas dataframe containing all the stock data, from oldest to newest
        stock_code: stock code of the required moving average
        column: column of the required moving average
        n: moving average of <n> elements
        skip_last: ignore the last <skip_last> records when computing moving average
    Returns
        An N-by-1 array of moving average
    
    """
    if skip_last != -1:
        target = stock_data[stock_code][column].values
    else:
        target = stock_data[stock_code][column].values[:-skip_last]
        
    return get_sliding_window(target, n).mean(axis=1).reshape(-1,1)
    
def get_lookback(stock_data, stock_code, column, n, skip_last = 0, **kwargs):
    """Get a lookback array
    Args:
        stock_data: pandas dataframe containing all the stock data, from oldest to newest
        stock_code: stock code of the required moving average
        column: column of the required moving average
        n: window size
        skip_last: ignore the last <skip_last> records when computing moving average
    Returns
        An N-by-window_size array of moving average
    
    """
    if skip_last != -1:
        target = stock_data[stock_code][column].values
    else:
        target = stock_data[stock_code][column].values[:-skip_last]
    return get_sliding_window(target, n)

def get_stock_data(stock_codes):
    """Retrieves data from a list of stock codes

    Args:
        stock_codes: <array_of_stock_codes_needed>

    Returns:
        a dict with stock codes as the keys and corresponding stock data as values

    """
    stock_data = {}
    for code in stock_codes:
        stock_data[code] = pd.read_csv("data/stock_prices/" + code + ".csv", index_col=0).iloc[::-1]

    return stock_data

def build_dataset(input_config, predict_n, training, stock_data, snake_size=10, previous=None):
    """Build dataset.

    Args:
        input_config: A input config dict.
            Format:
            {
                "stock_codes": <array_of_stock_codes_needed_to_build_the_dataset>,
                "stock_code": "predicting stock code",
                "column": "predicting value column name",
                "config": [
                    {"type": "feature type", <other_feature_configs},
                    {"type": "feature type", <other_feature_configs},
                    ...
                ]
            }
            Refer to train_models_sample.json.

        predict_n: Number of days of stock prices to predict
        training: True to get the training dataset, False to get the features for prediction.

    Returns:
        A tuple of m-data-by-n-features NumPy array and m-data-by-predict-n-labels NumPy array for training,
        or a tuple of m-data-by-t-timesteps-by-n-features NumPy array and m-data-by-predict-n-labels NumPy 
        array for training (if RNN/LSTM)
        or a 1-by-number-of-features NumPy array for prediction
    """
    
    # Get all the stock data
    #stock_data = get_stock_data(input_config["stock_codes"])
    if previous is not None:
        new_values = pd.DataFrame(previous.reshape(-1, 1), columns=[input_config["column"]])
        stock_data[input_config["stock_code"]] = stock_data[input_config["stock_code"]].append(new_values, ignore_index=True)

    print(stock_data[input_config["stock_code"]])

    target = stock_data[input_config["stock_code"]][input_config["column"]].values

    # Special case: index price
    if len(input_config["config"]) == 1 and input_config["config"][0]["type"] == "index_price":
        if training:
            x = np.arange(1, input_config["config"][0]["n"] + 1).reshape(-1, 1)
            y = target[-input_config["config"][0]["n"]:]
            return x, y
        else:
            x = np.arange(1, input_config["config"][0]["n"] + 1 + predict_n).reshape(-1, 1)
            y = target[-input_config["config"][0]["n"]:]
            return x
    
    # Build feature vectors by applying transformations on dataset 
    # specified in input_config
    transform = {
        "moving_avg": get_moving_avg,
        "lookback": get_lookback
    }
    
    if training:
        config_mapper = lambda config: transform[config["type"]](stock_data, skip_last=predict_n, **config)
    else:
        config_mapper = lambda config: transform[config["type"]](stock_data, **config)
    
    transformed_data = list(map(config_mapper, input_config["config"]))
    dataset_size = min(map(lambda arr: arr.shape[0], transformed_data))
    
    features = [ feature[-dataset_size:] for feature in transformed_data ]
    x = np.concatenate(features, axis=1)

    # Get a rolling time window if specified in config
    if "time_window" in input_config:
        time_window = input_config["time_window"]
        x = get_sliding_window(x, time_window)

    if training:
        output_shape = (x.shape[0], predict_n)
        y_size = output_shape[0] + predict_n - 1
        y = get_sliding_window(target[-y_size:], predict_n)

        return x, y
    
    else:
        output_shape = (x.shape[0], predict_n)
        y_size = output_shape[0] + predict_n - 1
        y = get_sliding_window(target[-y_size:], predict_n)

        # Get non-overlapping windows, aligning to the end
        x = x[::-1][:predict_n*(snake_size+1):predict_n][::-1]
        y = y[::-1][:predict_n*(snake_size):predict_n][::-1]
        return x
