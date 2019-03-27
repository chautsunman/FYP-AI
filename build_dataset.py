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

    #data_start = data.shape[0]
    #return np.lib.stride_tricks.as_strided(data[data_start:], shape, strides)
    return np.lib.stride_tricks.as_strided(data[:], shape, strides)

def get_moving_avg(stock_data, stock_code, column, n, skip_last = None, **kwargs):
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

    target = stock_data[stock_code][column].values

    if skip_last is not None:
        target = target[:-skip_last]

    return get_sliding_window(target, n).mean(axis=1).reshape(-1,1)

def get_lookback(stock_data, stock_code, column, n, skip=None, skip_last = None, **kwargs):
    """Get a lookback array
    Args:
        stock_data: pandas dataframe containing all the stock data, from oldest to newest
        stock_code: stock code of the required lookback array
        column: column of the required lookback array
        n: window size
        skip_last: ignore the last <skip_last> records when computing lookback array
    Returns
        An N-by-window_size array of lookback array

    """

    target = stock_data[stock_code][column].values

    if skip_last is not None:
        target = target[:-skip_last]

    if skip is not None:
        target = target[:-skip]

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

def normalize(data, input_options, normalize_type, normalize_data=None):
    """Normalize data."""

    if normalize_data is None:
        if normalize_type == "min_max":
            data_min = None
            data_max = None
            if "time_window" not in input_options:
                data_min = np.nanmin(data, axis=0)
                data_max = np.nanmax(data, axis=0)
                return (data - data_min) / (data_max - data_min), {"min": data_min.tolist(), "max": data_max.tolist()}
            else:
                data_shape = data.shape
                data = data.reshape(-1, data_shape[2])
                data_min = np.nanmin(data, axis=0)
                data_max = np.nanmax(data, axis=0)
                data = (data - data_min) / (data_max - data_min)
                data = data.reshape(-1, data_shape[1], data_shape[2])
                return data, {"min": data_min.tolist(), "max": data_max.tolist()}
    else:
        if normalize_type == "min_max":
            if "time_window" not in input_options:
                return (data - np.array(normalize_data["min"])) / (np.array(normalize_data["max"]) - np.array(normalize_data["min"]))
            else:
                data_shape = data.shape
                data = data.reshape(-1, data_shape[2])
                data = (data - np.array(normalize_data["min"])) / (np.array(normalize_data["max"]) - np.array(normalize_data["min"]))
                data = data.reshape(-1, data_shape[1], data_shape[2])
                return data

    return data

transform = {
    "moving_avg": get_moving_avg,
    "lookback": get_lookback
}

def build_training_dataset(input_options, predict_n, stock_data=None):
    """Build training dataset.

    Args:
        input_options: A input config dict.
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
        stock_data: Stock prices dictionary, same as output of get_stock_data.

    """

    if stock_data is None:
        # Get all the stock data
        stock_data = get_stock_data(input_options["stock_codes"])
    else:
        # copy the stock data
        stock_data_input = stock_data
        stock_data = {}
        for stock_code in stock_data_input:
            stock_data[stock_code] = stock_data_input[stock_code].copy()

    target = stock_data[input_options["stock_code"]][input_options["column"]].values

    other_data = {}

    # Special case: index price
    if len(input_options["config"]) == 1 and input_options["config"][0]["type"] == "index_price":
        x = np.arange(1, input_options["config"][0]["n"] + 1).reshape(-1, 1)
        y = target[-input_options["config"][0]["n"]:]
        return x, y, other_data

    # transform the data to features
    config_mapper = lambda config: transform[config["type"]](stock_data, skip_last=predict_n, **config)
    transformed_data = list(map(config_mapper, input_options["config"]))

    # set the dataset size as the minimum size of each feature
    dataset_size = min(map(lambda arr: arr.shape[0], transformed_data))

    # concatenate the features as the training set by aligning to the latest date
    features = [ feature[-dataset_size:] for feature in transformed_data ]
    x = np.concatenate(features, axis=1)

    # Get a rolling time window if specified in config
    if "time_window" in input_options:
        time_window = input_options["time_window"]
        x = get_sliding_window(x, time_window)

    # normalize features
    if "normalize" in input_options:
        x, normalize_data = normalize(x, input_options, input_options["normalize"])
        other_data["normalize_data"] = normalize_data

    # get the labels
    y = get_sliding_window(target, predict_n)[-x.shape[0]:]

    return x, y, other_data

def build_predict_dataset(input_options, predict_n, stock_data=None, predict=True, test_set='full', previous=None, skip_last=None):
    """Build prediction input.

    Args:
        input_options: A input config dict.
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
        stock_data: Stock prices dictionary, same as output of get_stock_data.
        predict: Whether to build the predict feature vector or the test set.
        previous: 1D NumPy array of previous predictions.
        skip_last: Number of rows to skip.

    """

    if stock_data is None:
        # Get all the stock data
        stock_data = get_stock_data(input_options["stock_codes"])
    else:
        # copy the stock data
        stock_data_input = stock_data
        stock_data = {}
        for stock_code in stock_data_input:
            stock_data[stock_code] = stock_data_input[stock_code].copy()

    if skip_last is not None:
        # skip the last n rows
        stock_data[input_options["stock_code"]] = stock_data[input_options["stock_code"]].iloc[:-skip_last, :]

    if previous is not None and previous.shape[0] > 0:
        # append the previous stock prices for using as if they are past data
        last_price = stock_data[input_options["stock_code"]][input_options["column"]].values[-1]
        new_values = None
        if previous.shape[0] > 1:
            new_values = pd.DataFrame({
                input_options["column"]: previous,
                "change": np.concatenate((
                    np.array([(previous[0] - last_price) / last_price]),
                    (previous[1:] - previous[:-1]) / previous[:-1]
                ))
            })
        else:
            new_values = pd.DataFrame({
                input_options["column"]: previous,
                "change": np.array([(previous[0] - last_price) / last_price])
            })
        stock_data[input_options["stock_code"]] = stock_data[input_options["stock_code"]].append(new_values, ignore_index=True)

    target = stock_data[input_options["stock_code"]][input_options["column"]].values

    # Special case: index price
    if len(input_options["config"]) == 1 and input_options["config"][0]["type"] == "index_price":
        x = np.arange(input_options["config"][0]["n"] + 1, input_options["config"][0]["n"] + 1 + predict_n).reshape(-1, 1)
        return x

    # transform the data to features
    if predict:
        config_mapper = lambda config: transform[config["type"]](stock_data, **config)
    else:
        config_mapper = lambda config: transform[config["type"]](stock_data, skip_last=predict_n, **config)
    transformed_data = list(map(config_mapper, input_options["config"]))

    # set the dataset size as the minimum size of each feature
    dataset_size = min(map(lambda arr: arr.shape[0], transformed_data))

    # concatenate the features as the training set by aligning to the latest date
    features = [ feature[-dataset_size:] for feature in transformed_data ]
    x = np.concatenate(features, axis=1)

    # Get a rolling time window if specified in config
    if "time_window" in input_options:
        time_window = input_options["time_window"]
        x = get_sliding_window(x, time_window)

    # normalize features
    if "normalize" in input_options:
        x = normalize(x, input_options, input_options["normalize"], input_options["normalize_data"])

    if predict:
        return np.expand_dims(x[-1], axis=0)
    else:
        # get the labels
        y = get_sliding_window(target, predict_n)[-x.shape[0]:]

        if test_set == "full":
            return x[-100:], y[-100:]
        elif test_set == "snakes":
            # Get non-overlapping windows, aligning to the end
            x_test = x[::-1][:100:10][::-1]
            y_test = y[::-1][:100:10][::-1]
            return x_test, y_test

def get_input_shape(input_options):
    """Returns the shape of the input as a tuple, can be
        1-D (for scikit-learn models & simple DNNs) or 2-D (for LSTM)
    """
    input_shape = [0]
    for config_item in input_options["config"]:
        if config_item["type"] == "lookback":
            input_shape[-1] += config_item["n"]
        elif config_item["type"] == "moving_avg":
            input_shape[-1] += 1
        elif config_item["type"] == "index_price":
            input_shape[-1] += config_item["n"]

    if "time_window" in input_options:
        input_shape.insert(0, input_options["time_window"])

    return tuple(input_shape)
