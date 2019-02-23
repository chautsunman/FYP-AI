import unittest

import numpy as np
import pandas as pd

from build_dataset import build_training_dataset, build_predict_dataset

stock_prices = {
    "GOOGL": pd.read_csv("test/GOOGL.csv", index_col=0).iloc[::-1]
}
prices = stock_prices["GOOGL"]["adjusted_close"].values

input_options = {
    "config": [
        {"type": "lookback", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"},
        {"type": "moving_avg", "n": 10, "stock_code": "GOOGL", "column": "adjusted_close"}
    ],
    "stock_codes": ["GOOGL"],
    "stock_code": "GOOGL",
    "column": "adjusted_close"
}

index_input_options = {
    "config": [
        {"type": "index_price", "n": 10}
    ],
    "stock_codes": ["GOOGL"],
    "stock_code": "GOOGL",
    "column": "adjusted_close"
}

class TestBuildTrainingDataset(unittest.TestCase):
    def test_btd_1(self):
        # input_options, predict 1

        x, y = build_training_dataset(input_options, 1, stock_data=stock_prices)

        self.assertEqual(x.shape, (3637, 11))
        self.assertEqual(y.shape, (3637, 1))

        x_start = [50.3228, 54.3227, 54.8694, 52.5974, 53.1641, 54.1221, 53.2393, 51.1629, 51.3435, 50.2802]
        x_end = [1097.99, 1125.89, 1118.62, 1141.42, 1151.87, 1122.89, 1105.91, 1102.38, 1102.12, 1127.58]
        x_start.append(sum(x_start) / len(x_start))
        x_end.append(sum(x_end) / len(x_end))
        self.assertEqual(x[0].tolist(), x_start)
        self.assertEqual(x[-1].tolist(), x_end)

        self.assertEqual(y[0][0], 50.9122)
        self.assertEqual(y[-1][0], 1128.63)

    def test_btd_2(self):
        # input_options, predict 10

        x, y = build_training_dataset(input_options, 10, stock_data=stock_prices)

        self.assertEqual(x.shape, (3628, 11))
        self.assertEqual(y.shape, (3628, 10))

        x_start = prices[:10].tolist()
        x_start.append(prices[:10].mean())
        x_end = prices[-20:-10].tolist()
        x_end.append(prices[-20:-10].mean())
        self.assertEqual(x[0].tolist(), x_start)
        self.assertEqual(x[-1].tolist(), x_end)

        self.assertEqual(y[0].tolist(), prices[10:20].tolist())
        self.assertEqual(y[-1].tolist(), prices[-10:].tolist())

    def test_btd_3(self):
        # index_input_options, predict 10

        x, y = build_training_dataset(index_input_options, 10, stock_data=stock_prices)

        self.assertEqual(x.tolist(), [[i] for i in range(1, 11)])
        self.assertEqual(y.tolist(), prices[-10:].tolist())

class TestBuildPredictDataset(unittest.TestCase):
    def test_bpd_1(self):
        # input_options, predict 1

        x = build_predict_dataset(input_options, 1, stock_data=stock_prices)

        self.assertEqual(x.shape, (1, 11))

        x_predict = prices[-10:].tolist()
        x_predict.append(prices[-10:].mean())
        self.assertEqual(x[0].tolist(), x_predict)

    def test_bpd_2(self):
        x = build_predict_dataset(input_options, 10, stock_data=stock_prices)

        self.assertEqual(x.shape, (1, 11))

        x_predict = [[1125.89, 1118.62, 1141.42, 1151.87, 1122.89, 1105.91, 1102.38, 1102.12, 1127.58, 1128.63]]
        x_predict[0].append(sum(x_predict[0]) / len(x_predict[0]))
        self.assertEqual(x.tolist(), x_predict)

    def test_bpd_3(self):
        # index_input_options, predict 10

        x = build_predict_dataset(index_input_options, 10, stock_data=stock_prices)

        self.assertEqual(x.tolist(), [[i] for i in range(11, 21)])

class TestBuildTestDataset(unittest.TestCase):
    def test_btd_1(self):
        # input_options, predict 1

        x, y = build_predict_dataset(input_options, 1, stock_data=stock_prices, predict=False)

        self.assertEqual(x.shape, (100, 11))
        self.assertEqual(y.shape, (100, 1))

        x_start = [1183.99, 1177.59, 1175.06, 1189.99, 1171.6, 1182.14, 1177.98, 1159.83, 1167.11, 1174.27]
        x_end = [1097.99, 1125.89, 1118.62, 1141.42, 1151.87, 1122.89, 1105.91, 1102.38, 1102.12, 1127.58]
        x_start.append(sum(x_start) / len(x_start))
        x_end.append(sum(x_end) / len(x_end))
        self.assertEqual(x[0].tolist(), x_start)
        self.assertEqual(x[-1].tolist(), x_end)

        self.assertEqual(y[0][0], 1191.57)
        self.assertEqual(y[-1][0], 1128.63)

    def test_btd_2(self):
        # input_options, predict 10

        x, y = build_predict_dataset(input_options, 10, stock_data=stock_prices, predict=False, snake_size=10)

        self.assertEqual(x.shape, (10, 11))
        self.assertEqual(y.shape, (10, 10))

        x_start = [1183.99, 1177.59, 1175.06, 1189.99, 1171.6, 1182.14, 1177.98, 1159.83, 1167.11, 1174.27]
        x_end = [1089.51, 1099.12, 1107.3, 1078.63, 1084.41, 1084, 1101.51, 1079.86, 1070.06, 1097.99]
        x_start.append(sum(x_start) / len(x_start))
        x_end.append(sum(x_end) / len(x_end))
        self.assertEqual(x[0].tolist(), x_start)
        self.assertEqual(x[-1].tolist(), x_end)

        y_start = stock_prices["GOOGL"]["adjusted_close"].values[-100:-90].tolist()
        y_end = stock_prices["GOOGL"]["adjusted_close"].values[-10:].tolist()
        self.assertEqual(y[0].tolist(), y_start)
        self.assertEqual(y[-1].tolist(), y_end)

if __name__ == '__main__':
    unittest.main()
