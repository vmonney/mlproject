from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


## Route for a home page
@app.route("/")
def home():
    return render_template("index.html")


## Route for a prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=int(request.form.get("reading_score")),
            writing_score=int(request.form.get("writing_score")),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("predict.html", results=results[0])
