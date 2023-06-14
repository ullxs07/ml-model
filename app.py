from __future__ import division, print_function
from flask import Flask,request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request, render_template
#from preprocessing import preprocess_image
from werkzeug.utils import secure_filename

# Load the trained model
#model = pickle.load(open('ullas1.pkl', 'rb'))

# Define a Flask app
app = Flask(__name__)

#model = load_model('please.h5')

#with open('ullas1.pkl', 'rb') as f:
#    model = pickle.load(f)

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
    
    # Add paths for other diseases

models = {}
for disease, model_path in model_paths.items():
    models[disease] = tf.keras.models.load_model(model_path)    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index2.html')
def page1():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return "No image file found."

    image_file = request.files['image']
    #image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Check if the file has a valid filename
    if image_file.filename == '':
        return "No image file selected."

    # Save the image to a desired location
    image_path = "uploads\\" + secure_filename(image_file.filename)
    image_file.save(image_path)

    disease = request.form['disease']

    if disease not in models:
        return jsonify({'error': 'Invalid disease'})
    


    # Load and preprocess the image
    # Load and preprocess the image
    #img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(124, 124))
    #img = cv2.imread(image_path)
    #img = Image.open(image_path)

    #img = cv2.resize(img, (124, 124))
   # img = tensorflow.keras.preprocessing.image.img_to_array(img)
    #img = np.expand_dims(img, axis=0)
    #img = img / 255.0
      # Convert BGR image to RGB
    #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the input shape of the model
    #resized_image = cv2.resize(rgb_image, (124, 124), interpolation=cv2.INTER_NEAREST)

    # Preprocess the image
import cv2
import numpy as np

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the image to (124, 124)
    resized_image = cv2.resize(image, (150, 150))
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    
    # Expand dimensions to match (1, 124, 124, 3)
    preprocessed_image = np.expand_dims(clahe_image, axis=-1)
    preprocessed_image = np.repeat(preprocessed_image, 3, axis=-1)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    return preprocessed_image
    
    processed_image = preprocess_image(image_path)

    model = models[disease]
    #predictions = model.predict(image)

# Add an extra dimension for the batch size
    #processed_image = np.expand_dims(processed_image, axis=0)


    result = model.predict(processed_image)

    # Convert predictions to a list
    prediction = result.tolist()

    # Return the predictions as a JSON response
    return jsonify(result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
