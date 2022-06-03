from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS


def prepare_image(img_file):
    image_bytes = img_file.read()
    pillow_img = Image.open(io.BytesIO(image_bytes))
    tr_img = np.asarray(pillow_img).astype(int)
    tr_img = tf.image.resize(tr_img, (32, 32))
    return tr_img[None,:,:,:]

model = tf.keras.models.load_model('Final_model.h5')


def predict_result(img):
    return "Cat" if model.predict(img) > 0.5 else "Dog"


app = Flask(__name__)
CORS(app)

@app.route("/predict", methods = ["POST"])

def predict_img():
    if "image" not in request.files:
        return "ERROR: NO IMAGE PROVIDED :<"

    image = request.files.get("image")

    if not image:
        return
    print(type(image))
    image = prepare_image(image)
    pred = predict_result(image)

    return jsonify({"prediction": pred})


if __name__ == '__main__':
    app.run(host='0.0.0.0')

