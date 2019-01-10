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

def build_dataset(input_config, predict_n, training):
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
        or a tuple of m-data-by-t-timesteps-by-n-features NumPy array and m-data-by-predict-n-labels NumPy array for training (if RNN/LSTM)
        or a 1-by-number-of-features NumPy array for prediction
    """

    stock_data = {}

    X = []

    for stock_code in input_config["stock_codes"]:
        stock_data[stock_code] = pd.read_csv("data/stock_prices/" + stock_code + ".csv", index_col=0).iloc[::-1]

    predict_data = stock_data[input_config["stock_code"]]
    predict_column = input_config["column"]
    predict_data_shape = predict_data.shape
    predict_data_len = predict_data_shape[0]
    y = pd.DataFrame(
        np.stack(predict_data[predict_column][i:i+predict_n] for i in range(predict_data_len - predict_n + 1)),
        index=predict_data.index[:predict_data_len - predict_n + 1],
        columns=["y_t+" + str(i) for i in range(predict_n)]
    )

    if training:
        # Training
        if len(input_config["config"]) == 1 and input_config["config"][0]["type"] == "index_price":
            x = np.arange(1, input_config["config"][0]["n"] + 1).reshape(-1, 1)
            y = y.iloc[-input_config["config"][0]["n"]:, 0].values
            return x, y

        for config in input_config["config"]:
            if config["type"] == "lookback":
                X.append(build_lookback(stock_data[config["stock_code"]][[config["column"]]], config["column"], config["n"]))

            elif config["type"] == "moving_avg":
                X.append(build_moving_avg(stock_data[config["stock_code"]][[config["column"]]], config["column"], config["n"]))

        data = y.join(X, how="inner")

        if "time_window" not in input_config:
            return data.iloc[:, predict_n:].values, data.iloc[:, :predict_n].values
        else:
            time_window = input_config["time_window"]
            x = data.iloc[:, predict_n:].values
            sample_size = x.shape[0]
            x = np.stack( [ x[i:i+time_window] for i in range(sample_size - time_window) ] )

            y = data.iloc[:, :predict_n].values[time_window:]

            return x, y

    else:
        # Prediction
        if len(input_config["config"]) == 1 and input_config["config"][0]["type"] == "index_price":
            return np.arange(
                input_config["config"][0]["n"] + 1,
                input_config["config"][0]["n"] + 1 + predict_n).reshape(-1, 1)

        if "time_window" not in input_config:
            for config in input_config["config"]:
                if config["type"] == "lookback":
                    X.append(stock_data[config["stock_code"]][config["column"]][-config["n"]:].values.reshape(1, -1))

                elif config["type"] == "moving_avg":
                    X.append(np.array([[stock_data[config["stock_code"]][config["column"]][-config["n"]:].sum() / config["n"]]]))

            return np.concatenate(X, axis=1)

        else:
            time_window = input_config["time_window"] 
            sample_size = stock_data[input_config["stock_code"]].shape[0]
            for config in input_config["config"]:
                if config["type"] == "lookback":
                    features = []
                    for i in range(sample_size - config["n"] - time_window + 1, sample_size - config["n"] + 1):
                        features.append(stock_data[config["stock_code"]][config["column"]][i:i+config["n"]].values.reshape(1, 1, -1))
                    X.append(np.concatenate(features, axis=1))

                elif config["type"] == "moving_avg":
                    features = []
                    for i in range(sample_size - config["n"] - time_window + 1, sample_size - config["n"] + 1):
                        features.append(np.array([[[stock_data[config["stock_code"]][config["column"]][i:i+config["n"]].sum() / config["n"]]]]))
                    X.append(np.concatenate(features, axis=1))

            return np.concatenate(X, axis=2)
