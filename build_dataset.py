import numpy as np
import pandas as pd

def build_moving_avg(data, column_name, lookback):
    moving_avg = data.cumsum()
    moving_avg = pd.concat([pd.DataFrame([0.0], index=["0"], columns=[column_name]), moving_avg])
    moving_avg.iloc[lookback + 1:, 0] = moving_avg.iloc[lookback:-1, 0].values - moving_avg.iloc[:-lookback - 1, 0].values
    moving_avg = moving_avg.iloc[lookback + 1:]
    moving_avg = moving_avg / lookback
    moving_avg = moving_avg.rename({column_name: "moving_avg"}, axis="columns")
    return moving_avg

def build_lookback(data, column_name, lookback):
    return pd.DataFrame(
        np.stack(data.loc[:, column_name][i:i+lookback] for i in range(0, data.loc[:, column_name].shape[0]-lookback)),
        index=data.index[lookback:],
        columns=["lookback_" + str(i) for i in range(lookback, 0, -1)]
    )

def build_dataset(input_config, training):

    stock_data = {}

    X = []

    for stock_code in input_config['stock_codes']:
        stock_data[stock_code] = pd.read_csv('data/stock_prices/' + stock_code + '.csv', index_col=0).iloc[::-1]

    y = stock_data[input_config["stock_code"]][[input_config["column"]]]

    if training:
        # Training
        if len(input_config["config"]) == 1 and input_config["config"][0]["type"] == "index_price":
            x = np.arange(1, input_config["config"][0]["n"] + 1)
            y = y.iloc[-input_config["config"][0]["n"]:, 0].values
            return x, y

        for config in input_config['config']:
            if config['type'] == 'lookback':
                X.append(build_lookback(stock_data[config['stock_code']][[config["column"]]], config["column"], config["n"]))

            if config['type'] == 'moving_avg':
                X.append(build_moving_avg(stock_data[config['stock_code']][[config["column"]]], config["column"], config["n"]))

        data = y.join(X, how="inner")

        return data.iloc[:, 1:].values, data.iloc[:, 0].values

    else:
        # Prediction
        for config in input_config['config']:
            n = config['n']

            stock_code = config['stock_code']
            column = config['column']

            if config['type'] == 'lookback':
                X.append(np.array([ np.sum(stock_data[stock_code].loc[-n:, column]) / n ]))

            if config['type'] == 'moving_avg':
                X.append((stock_data[stock_code].loc[-3:, column], n))


        input_len = map(lambda arr: arr.shape[1], X)
        min_input_len = reduce(lambda a, b: min(a, b), input_len)

        return np.concatenate(X).ravel()

