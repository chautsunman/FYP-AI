import pandas as pd

def build_moving_avg(data, lookback=1):
    ret = np.cumsum(data, dtype=float)
    ret[lookback:] = ret[lookback:] - ret[:-lookback]
    result = ret[lookback - 1:-1] / lookback

    return np.reshape(result, (result.shape[0], 1))

def build_lookback(data, lookback=1):
    return np.stack(data[i:i+lookback] for i in range(0, data.shape[0]-lookback))

def build_dataset(input_config, training):

    stock_data = {}

    X = []

    for stock_code in input_config['stock_codes']:     
        stock_data[stock_code] = pd.read_csv('data/stock_prices/' + stock_code + '.csv').iloc[::-1]

    if training: 
        # Training
        for config in input_config['config']:

            n = config['n']

            stock_code = config['stock_code']
            column = config['column']:

                if config['type'] == 'lookback':
                    X.append(build_lookback(stock_data[stock_code].loc[:, column], n))

            if config['type'] == 'moving_avg':
                X.append(build_moving_avg(stock_data[stock_code].loc[:, column], n))

        return np.concatenate(x[0:min_input_len,:] for x in X, axis=1), stock_data.loc[-min_input_len:, input_config['column']]

    else:
        # Prediction
        for config in input_config['config']:
            n = config['n']

            stock_code = config['stock_code']
            column = config['column']:

            if config['type'] == 'lookback':
                X.append(np.array([ np.sum(stock_data[stock_code].loc[-n:, column]) / n ]))

        if config['type'] == 'moving_avg':
            X.append((stock_data[stock_code].loc[-3:, column], n))


        input_len = map(lambda arr: arr.shape[1], X)
        min_input_len = reduce(lambda a, b: min(a, b), input_len)

        return np.concatenate(X).ravel()

