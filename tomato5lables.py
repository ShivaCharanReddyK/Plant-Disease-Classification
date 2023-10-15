from flask import Flask, request
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

MODEL = tf.keras.models.load_model(
    "tomato5.h5")
    

CLASS_NAMES = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato_healthy']

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")


@app.route("/ping", methods=["GET"])
def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(
        Image.open(BytesIO(data)).convert(
            "RGB").resize((256, 256))  # image resizing
    )

    return image


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=True
    )
