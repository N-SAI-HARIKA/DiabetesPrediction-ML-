from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained ML model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data
        data = request.json
        features = [
            data.get("pregnancies", 0),
            data.get("glucose", 0),
            data.get("bloodPressure", 0),
            data.get("skinThickness", 0),
            data.get("insulin", 0),
            data.get("bmi", 0.0),
            data.get("dpf", 0.0),
            data.get("age", 0),
        ]

        # Convert to numpy array and predict
        input_data = np.array([features], dtype=float)
        prediction = model.predict(input_data)[0]

        # Generate result message
        message = "You are likely to have diabetes." if prediction == 1 else "You are unlikely to have diabetes."
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"message": "An error occurred during prediction.", "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
