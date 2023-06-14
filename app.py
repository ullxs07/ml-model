from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

model_paths = {
    'ARMD': 'ARM.h5',
    'BRVO': 'BRVO.h5',
    'CRS': 'CRS.h5',
    'CSR': 'CSR.h5',
    'CRVO': 'CRVO.h5',
    'DR': 'DR.h5',
    'DN': 'DN.h5',
    'LS': 'LS.h5',
    'MH': 'MH.h5',
    'MYA': 'MYA.h5',
    'ODE': 'ODE.h5',
    'ODP': 'ODP.h5',
    'RPEC': 'RPEC.h5',
    'RS': 'RS.h5',
    'TSLN': 'TSLN.h5',
    'ODC': 'ODC.h5',
}

models = {}
for disease, model_path in model_paths.items():
    models[disease] = load_model(model_path)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (150, 150))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    preprocessed_image = np.expand_dims(clahe_image, axis=-1)
    preprocessed_image = np.repeat(preprocessed_image, 3, axis=-1)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    return preprocessed_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index2.html')
def page1():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image file found."

    image_file = request.files['image']

    if image_file.filename == '':
        return "No image file selected."

    image_path = "uploads/" + secure_filename(image_file.filename)
    image_file.save(image_path)

    disease = request.form['disease']

    if disease not in models:
        return jsonify({'error': 'Invalid disease'})

    processed_image = preprocess_image(image_path)

    model = models[disease]
    result = model.predict(processed_image)
    prediction = result.tolist()

    return jsonify(result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
