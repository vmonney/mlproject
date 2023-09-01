import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        tuned_models = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}")

            grid_search = GridSearchCV(
                model,
                param_grid=param[model_name],
                cv=3,
            )
            grid_search.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            tuned_models[model_name] = r2_score(y_test, y_pred)

        return tuned_models

    except Exception as e:
        raise CustomException(e)
