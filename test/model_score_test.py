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
            Model score ranged [0...10]
            
    """
    def test_best_case(self):
        """
            Test the best case (when prediction is perfectly correct)
            score should be equal to 10
        """
        # Generate [ 1...101 ]
        actual = np.arange(1, 102).tolist()

        # Generate [2...11], [12...21], ..., [92...101]
        predicted = np.arange(2, 102).reshape(10, 10).tolist()

        time_interval = 10
        self.assertAlmostEqual(model_rating(actual, predicted, time_interval), 1)

    def test_worst_case(self):
        """
            Test the worst case (when prediction is perfectly wrong)
            Score should be 10
        """
        # Generate [ 1...101 ]
        actual = np.arange(2000, 2101).tolist()

        # Generate [ 1000...1009 ], [ 1010...1019 ]..., [ 1091...1099 ]
        predicted = np.arange(1000, 1100).reshape(10, 10).tolist()

        time_interval = 10
        self.assertEqual(model_rating(actual, predicted, time_interval), 0)

    def test_correct_over(self):
        """
            Test the case when the direction is correct but magnitude is
            overestimated
        """
        # Generate [ 1...101 ]
        actual = np.arange(1, 102).tolist()

        # Generate [ 2...11 ], [ 12...21 ], ..., [ 92...101 ] * 100.05%
        predicted = (np.arange(2, 102).reshape(10, 10) * 1.005).tolist()

        time_interval = 10
        self.assertAlmostEqual(model_rating(actual, predicted, time_interval), 0.811605)

    def test_correct_under(self):
        """
            Test the case when the direction is correct but magnitude is
            underestimated 
        """
        # Generate [ 101...1 ]
        actual = np.arange(1, 102).tolist()

        # Generate [ 2...11 ], [ 12...21 ], ..., [ 92...101 ] * 99.5%
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
