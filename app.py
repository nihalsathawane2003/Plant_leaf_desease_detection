from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\Nihal\\Desktop\\Plant leaf detection\\Backend\\plant_disease_model.keras')

# Define class names based on your model
class_names = ['Healthy', 'Powdery', 'Rust',  # Adjust according to your classes
               # Add more class names as needed
              ]

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Preprocess the image and perform prediction
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]

    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
