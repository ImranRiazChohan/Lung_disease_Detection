from flask import Flask, jsonify, request
from tensorflow import keras
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from skimage.metrics import structural_similarity
import cv2

app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('model.h5')

# Define the class names
class_names = ["Tuberculosis", "healthy", "latent-tb", "uncertain-tb"]

def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))  # Resize image to match the input size used during training
    image_array = np.asarray(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def resize_and_gray(image):
    resized = cv2.resize(image, (800, 800))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray

def check_similarity(image1, image2):
    score, _ = structural_similarity(image1, image2, full=True)
    return score

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains a file or a URL
        if 'file' in request.files:
            # Get the image file from the request
            image_file = request.files['file']
            # Read image bytes and preprocess
            image_bytes = image_file.read()
        elif 'url' in request.form:
            # Get the image URL from the request
            image_url = request.form['url']
            # Download image from the URL and preprocess
            response = requests.get(image_url)
            response.raise_for_status()  # Check for HTTP errors
            image_bytes = response.content
        else:
            raise ValueError("Neither file nor URL provided in the request.")

        input_image_array = preprocess_image(image_bytes)

        # Load the reference image for similarity check
        reference_image = cv2.imread('disease.jpg')
        reference_image_gray = resize_and_gray(reference_image)

        # Resize and convert the input image to grayscale for similarity check
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_image_gray = resize_and_gray(np.asarray(input_image))

        # Check similarity
        similarity_score = check_similarity(reference_image_gray, input_image_gray)

        # If similarity is above a certain threshold, proceed with prediction
        if similarity_score >= 0.9:
            # Make prediction
            predictions = model.predict(input_image_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]

            # Return the result as JSON
            return jsonify({'class': predicted_class})
        else:
            return jsonify({'error': 'Image not valid for this model. Similarity score too low.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)