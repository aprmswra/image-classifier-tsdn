from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.utils import  load_img, img_to_array 
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

CLASSES = ['Bacterial leaf blight', 'Brown spot', 'Healthy', 'Leaf smut']
MODEL_PATH = 'rice_model_desease.hdf5'
model = load_model(MODEL_PATH)
model.make_predict_function()

IMAGE_DIM = 240,240
def model_predict(img_path, model):
    img_array = cv2.imread(img_path)
    img_array = cv2.resize(img_array, IMAGE_DIM)
    im_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    x = np.array(im_rgb).reshape(*IMAGE_DIM,3)/255.0
    x = np.expand_dims(x, 0)
    preds = model.predict(x)
    class_id = np.argmax(preds)
    label =  CLASSES[class_id]
    return label

@app.route('/Image_Classification', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, '/', secure_filename(f.filename))
    f.save(file_path)
    label = model_predict(file_path, model)
    result = label
    print(f"result: {result}")
    return result

@app.route('/')
def home():
    return "Server is running!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)