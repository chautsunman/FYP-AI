import unittest
import numpy as np
from rating_calculation import model_rating


class TestModelScore(unittest.TestCase):
    """
        rating_calculation expected input/output:
        Input:
            actual: 
            predicted: (n,time_interval) 2d-list of
        Output:
            10000 * sum(
            
    """
    def test_best_case(self):
        # Generate [ 1...101 ]
        actual = np.arange(1, 102).tolist()

        # Generate [2...11], [12...21], ..., [92...101]
        predicted = np.arange(2, 102).reshape(10, 10).tolist()

        time_interval = 10
        self.assertEqual(model_rating(actual, predicted, time_interval), 1)

    def test_worst_case():
        # Generate [ 1...101 ]
        actual = np.arange(1, 102).tolist()

        # Generate [2*1.01...11*1.01], [12*1.01...21*1.01], ..., [92*1.01...101*1.01]
        predicted = (np.arange(2, 102).reshape(10, 10) * 1.01).tolist()

        time_interval = 10
        self.assertEqual(model_rating(actual, predicted, time_interval, 0))

    """
        Assume alpha = 0.2,
        Test the score when MAE=0.005
    """
    def test_correct_over(self):
        # Generate [ 1...101 ]
        actual = np.arange(1, 102).tolist()

        # Generate [ 2...11 ], [ 12...21 ], ..., [ 92...101 ]
        predicted = (np.arange(2, 102).reshape(10, 10) * 1.005).tolist()

        time_interval = 10
        self.assertAlmostEqual(model_rating(actual, predicted, time_interval), 0.811605)

    def test_correct_under(self):
        # Generate [ 101...1 ]
        actual = np.arange(1, 102).tolist()

        # Generate [ 2...11 ], [ 12...21 ], ..., [ 92...101 ]
        predicted = (np.arange(2, 102).reshape(10, 10) * 0.995).tolist()

        time_interval = 10
        self.assertAlmostEqual(model_rating(actual, predicted, time_interval), 0.851605)

    """
    def test_incorrect_direction():
        # Generate [ 1000.01, 1000.02, ..., 1001.01 ]
        actual = (np.arange(1, 102) / 100 + 1000).tolist()

        # Generate [ 1001.01, ...
        predicted = (np.arange(101, 1, -1) / 100 + 1000).reshape(10, 10).tolist()

        time_interval = 10
        self.assertAlmostEqual(model_rating(actual, predicted, time_interval))
    """
