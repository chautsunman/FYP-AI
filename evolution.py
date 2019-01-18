import numpy as np
from sklearn.model_selection import train_test_split

from build_dataset import build_dataset

def evolution(ModelClass, iterations):
    """Evolution algorithm."""

    # initialize some random models
    models = ModelClass.random_models(10)

    errors = []

    for i in range(iterations):
        errors = []

        for model in models:
            # prepare the data
            x, y = build_dataset(model.input_options, model.model_options["predict_n"], True)
            # split the data into training set and testing set
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            # train the model
            model.train(x_train, y_train)
            # calculate the model error
            y_predict = model.predict(x_test)
            errors.append(model.error(y_test, y_predict))

        # select top models
        error_idx_sorted = np.argsort(errors)
        top_models = [models[i] for i in error_idx_sorted[:2]]

        # cross-over models and breed new models
        if i < iterations - 1:
            models = ModelClass.evolve(top_models, 10)

    # return the best model
    best_model_idx = np.argmin(errors)
    best_model = models[best_model_idx]
    return best_model
