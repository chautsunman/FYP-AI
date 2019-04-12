import argparse
import json

from train_models import train_models

from models.linear_index_regression import LinearIndexRegression
from models.svr_index_regression import SupportVectorIndexRegression
import math
import numpy as np

# Calculate the MAE of the prediction from the model
# Assume following format:
# actual prices = [price_0, price_1, ..., price_n] # n+1 day in total
# predicted prices = [price_1, ..., price_n] # (snakes), n day in total
def relative_mean_absolute_error(actual_prices, predicted_prices, t_0, time_interval):
    rmae = 0

    for i in range(time_interval):
        rmae += abs(predicted_prices[t_0 + i] / actual_prices[t_0 + i + 1] - 1)
    rmae /= time_interval

    if rmae < 0:
        print(
            "error: [relative_mean_absolute_error Line 29] RMAE cannot be smaller than zero")
        exit(-1)
    return rmae


def isSameDirection(actual_prices, predicted_prices, t_0, time_interval):
    return (actual_prices[t_0 + time_interval] - actual_prices[t_0] < 0) == (predicted_prices[t_0 + time_interval - 1] - actual_prices[t_0] < 0)


def isUnderestimated(actual_prices, predicted_prices, t_0, time_interval):
    # given that they are same direction
    return abs(actual_prices[t_0 + time_interval] - actual_prices[t_0]) >= abs(predicted_prices[t_0 + time_interval - 1] - actual_prices[t_0])


def percentageChange(today_price, predicted_price):
    return (predicted_price / today_price - 1)


# f(e)
def model_scoring_func(error_rate, sd):
    # return ((error_rate - 0.1)/0.1)**4 if error_rate < 0.1 else 0
    return ((error_rate - sd)/sd)**4 if error_rate < sd else 0

# Calculate the rating based on RMAE


def model_rating(actual_prices, snakes, time_interval, sd):

    predicted_prices = []

    for sublist in snakes:
        for item in sublist:
            predicted_prices.append(item)

    alpha = 0.2
    if predicted_prices == []:
        print(
            "[model_rating] predicted prices with a length of zero")
        return 0

    if len(actual_prices) - 1 != len(predicted_prices):
        print(
            "error: [model_rating] predicted price length inequal to actual price length")
        exit(-1)

    if len(predicted_prices) % time_interval != 0:
        # if length of predicted prices is not divisible by the time_interval, it is an error
        print(
            "error: [model_rating] predicted price length not divisible by time interval")
        exit(-1)

    rating = 0

    for i in range(len(predicted_prices)//time_interval):
        error_rate = relative_mean_absolute_error(
            actual_prices, predicted_prices, i*time_interval, time_interval)

        if not isSameDirection(actual_prices, predicted_prices, i*time_interval, time_interval):
            reward = 0
        elif isUnderestimated(actual_prices, predicted_prices, i*time_interval, time_interval):
            reward = 1.0
        else:
            reward = 0.8

        rating += (1 - alpha) * model_scoring_func(error_rate, sd) + alpha * reward

    return rating / (len(predicted_prices) / time_interval)


def calculate_traffic_light_score(models, sd, VAILD_MODEL_THRESHOLD):
    traffic_light_score = 0
    counter = 0

    for i in models:
        if i["score"] < VAILD_MODEL_THRESHOLD:
            continue

        if i["percentageChange"] > 0:
            traffic_light_score += i["score"] * theta(i["percentageChange"], sd)
        else:
            traffic_light_score -= i["score"] * theta(i["percentageChange"], sd)

        counter +=1

    return (traffic_light_score/counter) if counter > 0 else 0

def theta(percentageChange, sd):
    return math.expm1(100*abs(percentageChange)) / math.expm1(100*sd)

def calculate_trend_score(predictions, prices):
    score = 0
    for i in range(10):
        predictions_direction = np.sign(predictions[i + 1:] - predictions[:-i - 1])
        prices_direction = np.sign(prices[i + 1:] - prices[:-i - 1])
        score += np.sum(np.where(predictions_direction == prices_direction, 1, 0)) / predictions.shape[0]
    return score / 10

def count_trend(predictions, last_price):
    trends = np.where(predictions - last_price >= 0, 1, -1)
    return 1 if np.sum(np.where(trends == 1, 1, 0)) >= predictions.shape[0] / 2 else -1

def calculate_stock_trend_score(models, accurate_threshold):
    score = 0
    accurate_models = 0
    for model in models:
        if model["trendScore"] >= accurate_threshold:
            score += model["trendScore"] * model["trend"]
            accurate_models += 1
    return score / accurate_models if accurate_models > 0 else 0
