from flask import Flask, render_template, request, redirect, session
import pandas as pd
from aml_model import predict

app = Flask(__name__)
app.secret_key = "aml-demo-secret"

# ---- SIMPLE LOGIN ----
USERNAME = "admin"
PASSWORD = "admin123"


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (
            request.form.get("username") == USERNAME
            and request.form.get("password") == PASSWORD
        ):
            session["logged_in"] = True
            return redirect("/upload")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not session.get("logged_in"):
        return redirect("/")

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("upload.html", error="Please upload a CSV file")

        df = pd.read_csv(file)

        # PURE INFERENCE
        preds, probs = predict(df)

        df["Fraud_Prediction"] = preds
        df["Fraud_Probability"] = probs

        return render_template(
            "results.html",
            table=df.head(30).to_html(index=False),
            total=len(df),
            fraud_count=int(preds.sum()),
        )

    return render_template("upload.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
