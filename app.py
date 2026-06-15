from pathlib import Path
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"

FEATURE_ORDER = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Self_Employed",
    "Married",
    "Dependents",
    "Education",
    "Property_Area",
]

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "ApplicantIncome": float(request.form.get("ApplicantIncome", 0)),
            "CoapplicantIncome": float(request.form.get("CoapplicantIncome", 0)),
            "LoanAmount": float(request.form.get("LoanAmount", 0)),
            "Loan_Amount_Term": float(request.form.get("Loan_Amount_Term", 0)),
            "Credit_History": float(request.form.get("Credit_History", 0)),
            "Self_Employed": request.form.get("Self_Employed", "No"),
            "Married": request.form.get("Married", "No"),
            "Dependents": request.form.get("Dependents", "0"),
            "Education": request.form.get("Education", "Graduate"),
            "Property_Area": request.form.get("Property_Area", "Urban"),
        }

        features =pd.DataFrame([data])
        # [[data[feature] for feature in FEATURE_ORDER]]
        probability = model.predict_proba(features)[0][1]
        prediction = {
            "approved": probability >= 0.5,
            "probability": round(probability * 100, 2),
        }

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
