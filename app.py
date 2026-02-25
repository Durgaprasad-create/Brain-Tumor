from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load model dictionary
MODEL_PATH = "model/brain_tumor_model.pkl"
model_data = pickle.load(open(MODEL_PATH, "rb"))
model = model_data["model"]
model_accuracy = model_data["accuracy"]


def predict_tumor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid image file.")

    img = cv2.resize(img, (64, 64))
    img = img.flatten() / 255.0   # must match training
    img = img.reshape(1, -1)

    prediction = model.predict(img)[0]
    confidence = model.predict_proba(img)[0][prediction]

    return prediction, round(confidence * 100, 2)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    suggestion = ""
    confidence = None

    if request.method == "POST":

        file = request.files.get("mri")

        if not file or file.filename == "":
            return render_template("index.html")

        save_path = os.path.join("static", "uploaded.jpg")
        file.save(save_path)

        prediction, confidence = predict_tumor(save_path)

        if prediction == 1:
            result = "🧠 Brain Tumor Detected"
            suggestion = "Consult a neurologist immediately."
        else:
            result = "✅ No Tumor Detected"
            suggestion = "Maintain regular health checkups."

    return render_template(
        "index.html",
        result=result,
        suggestion=suggestion,
        confidence=confidence,
        model_accuracy=round(model_accuracy * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)