import numpy as np
import pandas as pd

def build_moving_avg(data, column_name, lookback):
    """Builds moving average dataset.

    Args:
        data: A Pandas DataFrame (1 column (<column_name>), sorted from oldest to latest).
        column_name: Name of the data column used to calculate the moving average.
        lookback: The number of data used to calculate the moving average.

    Returns:
        A Pandas DataFrame with moving_avg column.
    """

    moving_avg = data.cumsum()
    moving_avg = pd.concat([pd.DataFrame([0.0], index=["0"], columns=[column_name]), moving_avg])
    moving_avg.iloc[lookback + 1:, 0] = moving_avg.iloc[lookback:-1, 0].values - moving_avg.iloc[:-lookback - 1, 0].values
    moving_avg = moving_avg.iloc[lookback + 1:]
    moving_avg = moving_avg / lookback
    moving_avg = moving_avg.rename({column_name: "moving_avg"}, axis="columns")
    return moving_avg

def build_lookback(data, column_name, lookback):
    """Builds lookback dataset.

    Args:
        data: A Pandas DataFrame (sorted from oldest to latest).
        column_name: Name of the data column used to build the lookback.
        lookback: The number of lookback data.

    Returns:
        A Pandas DataFrame with <lookback> columns.
    """

    return pd.DataFrame(
        np.stack(data.loc[:, column_name][i:i+lookback] for i in range(0, data.loc[:, column_name].shape[0]-lookback)),
        index=data.index[lookback:],
        columns=["lookback_" + str(i) for i in range(lookback, 0, -1)]
    )

def get_feature_vector_size(input_config):
    """Get the size of the feature vector based on the input config
    
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

    Returns:
        number of features of the feature_vector.

    """
    feature_vector_size = 0
    for config in input_config["config"]:
        if config["type"] == "lookback" or config["type"] == "index_price":
            feature_vector_size += config["n"]

        elif config["type"] == "moving_avg":
            feature_vector_size += 1

    return feature_vector_size


def get_feature_vector(input_config, previous_prices, last_feature_vector):
    """Get a new feature vector based on the last one and the input config

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
        last_feature_vector:  The last feature vector

    Returns:
        A 1-by-number-of-features NumPy array for prediction.

    """ 

    stock_data = {}
    
    # Read the stock prices
    for stock_code in input_config["stock_codes"]:
        stock_data[stock_code] = pd.read_csv('data/stock_prices/' + stock_code + '.csv', index_col=0).iloc[::-1]


    feature_vector = []

    for config in input_config["config"]:
        if config["type"] == "lookback":
            # Determine how many prices should we get from dataset
            # If dataset contains only data up to time t
            # e.g. if lookback = 10, to predict price at t + 4
            # We first predict prices from t + 1 to t + 3
            # Then grab the prices from t - 6 to t (last 7 records)   
            # and add the predictions from t + 1 to t + 3 to the array
            # Hence we want to grab the -10 + 3 = -7 = last 7 records  
            n = -config["n"] + len(previous_prices)
            column = config["column"]

            if n < 0:
                feature_vector.extend(stock_data[stock_code][column][n:])

            feature_vector.extend(previous_prices[-min(config["n"], len(previous_prices)):])

        elif config["type"] == "moving_avg":
            # Follow same logic as above
            n = -config["n"] + len(previous_prices)
            column = config["column"]
            cumsum = 0
            if n < 0:
                cumsum += sum(stock_data[stock_code][column][n:])

            cumsum += sum(previous_prices[-min(config["n"], len(previous_prices)):])
            moving_avg = cumsum / config["n"]

            feature_vector.append(moving_avg)

        """
        elif config["type"] == "index_price": 
            # Length of new feature vector should be the same as the old one
            # The starting index for index_price list should be same as the
            # next index of the feature vector built up to this point
            start_pos = len(feature_vector)
            end_pos = start_pos + config["n"]
            feature_vector.extend(last_feature_vector[start_pos:end_pos])
            feature_vector.append(last_feature_vector[end_pos-1] + 1)
        """

    return np.array(feature_vector).reshape(1, -1)


def build_dataset(input_config, training):
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
        training: True to get the training dataset, False to get the features for prediction.

    Returns:
        A tuple of m-data-by-n-features NumPy array and m-data labels NumPy array for training,
        or a 1-by-number-of-features NumPy array for prediction.
    """

    stock_data = {}

    X = []

    for stock_code in input_config['stock_codes']:
        stock_data[stock_code] = pd.read_csv('data/stock_prices/' + stock_code + '.csv', index_col=0).iloc[::-1]

    y = stock_data[input_config["stock_code"]][[input_config["column"]]]

    if training:
        # Training
        if len(input_config["config"]) == 1 and input_config["config"][0]["type"] == "index_price":
            x = np.arange(1, input_config["config"][0]["n"] + 1).reshape(-1, 1)
            y = y.iloc[-input_config["config"][0]["n"]:, 0].values
            return x, y

        for config in input_config['config']:
            if config['type'] == 'lookback':
                X.append(build_lookback(stock_data[config['stock_code']][[config["column"]]], config["column"], config["n"]))

            elif config['type'] == 'moving_avg':
                X.append(build_moving_avg(stock_data[config['stock_code']][[config["column"]]], config["column"], config["n"]))

        data = y.join(X, how="inner")

        return data.iloc[:, 1:].values, data.iloc[:, 0].values

    else:
        # Prediction
        if len(input_config["config"]) == 1 and input_config["config"][0]["type"] == "index_price":
            predict_n = input_config["config"][0]["predict_n"] if "predict_n" in input_config["config"][0] else 1
            return np.arange(
                input_config["config"][0]["n"] + 1,
                input_config["config"][0]["n"] + 1 + predict_n).reshape(-1, 1)

        for config in input_config['config']:
            if config['type'] == 'lookback':
                X.append(stock_data[config["stock_code"]][config["column"]][-config["n"]:].values.reshape(1, -1))

            elif config['type'] == 'moving_avg':
                X.append(np.array([[stock_data[config["stock_code"]][config["column"]][-config["n"]:].sum() / config["n"]]]))

        return np.concatenate(X, axis=1)
