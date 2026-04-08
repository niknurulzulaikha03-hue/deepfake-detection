from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Upload size limit (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Load trained model
model = load_model("model/deepfake_model.h5")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():

    result = ""
    confidence = 0

    if request.method == "POST":

        file = request.files["video"]

        if file.filename == "":
            result = "No file selected"
            return render_template("index.html", result=result)

        # Check file type
        if not allowed_file(file.filename):
            result = "Only MP4, AVI, MOV allowed"
            return render_template("index.html", result=result)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        # Preprocess video
        features = preprocess_video(filepath)

        if features is None:
            result = "Error processing video"
            return render_template("index.html", result=result)

        features = np.expand_dims(features, axis=0)

        # Predict
        prediction = model.predict(features, verbose=0)

        real_prob = prediction[0][0]
        fake_prob = prediction[0][1]

        if fake_prob > real_prob:
            result = "Fake Video"
            confidence = fake_prob * 100
        else:
            result = "Real Video"
            confidence = real_prob * 100

    return render_template("index.html",
                           result=result,
                           confidence=confidence)


if __name__ == "__main__":
    app.run(debug=False)
