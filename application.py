from flask import Flask, request, render_template, redirect, flash
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

applicationlication = Flask(__name__)
application = applicationlication


## Route for a home page
@application.route("/")
def home():
    return render_template("index.html")


## Route for a prediction page
@application.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        # Fetch form data
        gender = request.form.get("gender")
        ethnicity = request.form.get("ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")

        reading_score_str = request.form.get("reading_score")
        writing_score_str = request.form.get("writing_score")

        if not reading_score_str or not writing_score_str:
            flash("Scores should be provided!")
            return redirect(request.url)

        try:
            reading_score = int(reading_score_str)
            writing_score = int(writing_score_str)
        except (TypeError, ValueError):
            flash("Scores should be valid integers!")
            return redirect(request.url)

        # Validate form data
        valid_genders = ["male", "female"]
        if gender not in valid_genders:
            flash("Invalid gender selected!")
            return redirect(request.url)

        valid_ethnicities = ["group B", "group C", "group A", "group D", "group E"]
        if ethnicity not in valid_ethnicities:
            flash("Invalid ethnicity selected!")
            return redirect(request.url)

        valid_parental_level_of_education = [
            "bachelor's degree",
            "some college",
            "master's degree",
            "associate's degree",
            "high school",
            "some high school",
        ]
        if parental_level_of_education not in valid_parental_level_of_education:
            flash("Invalid parental level of education selected!")
            return redirect(request.url)

        valid_lunch = ["standard", "free/reduced"]
        if lunch not in valid_lunch:
            flash("Invalid lunch selected!")
            return redirect(request.url)

        valid_test_preparation_course = ["none", "completed"]
        if test_preparation_course not in valid_test_preparation_course:
            flash("Invalid test preparation course selected!")
            return redirect(request.url)

        if not (0 <= reading_score <= 100) or not (0 <= writing_score <= 100):
            flash("Scores should be between 0 and 100!")
            return redirect(request.url)

        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("predict.html", results=results[0])


if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=True)
