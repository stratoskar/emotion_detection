# Standard imports
from flask import Flask, render_template, request, redirect, url_for # Build a web application
from tensorflow.keras.models import load_model # Use a deep learning model
from tensorflow.keras.preprocessing import image
import numpy as np
import os # Interract with the os
from werkzeug.utils import secure_filename
import cv2
import json  # Import the json library

app = Flask(__name__)

# Configure upload folder where images will be saved
UPLOAD_FOLDER = 'uploads'

# Configure allowed extentions of images the user can upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

# Load the trained model (using the model.py code file)
model_path = 'models/emotion_recognition_model_fer.h5'
try:
    model = load_model(model_path)

    # classification outcomes
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','neutral']
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading the model: {e}")
    model = None
    emotion_labels = []


# Define the target image size used during training
img_width, img_height = 48, 48

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    try:
        # Load the image using OpenCV for more flexibility
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        # Resize the image
        img_resized = cv2.resize(img, (img_width, img_height),
                                  interpolation=cv2.INTER_AREA)  # Use INTER_AREA for shrinking

        # Convert to numpy array and expand dimensions
        img_array = np.expand_dims(img_resized, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0

        return img_array

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html',
                              error="Model not loaded. Please check the server.")

    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            # Construct the full filepath
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Save the uploaded file
            file.save(filepath)

            # Preprocess the image
            processed_image = preprocess_image(filepath)

            # Handle preprocessing errors
            if processed_image is None:
                os.remove(filepath)  # Clean up the saved file
                return render_template('index.html',
                                      error='Error processing the uploaded image.')

            # Make the prediction
            predictions = model.predict(processed_image)
            # Get the predicted class
            predicted_class_index = np.argmax(predictions)
            # Map the class index to the emotion label
            predicted_emotion = emotion_labels[predicted_class_index]

            # Prepare data for the execution plan
            execution_plan_data = {
                "original_image_path": filepath,
                "preprocessed_shape": processed_image.shape,
                "model_architecture": "Robust CNN", 
                "predicted_emotion": predicted_emotion,
                "all_emotions_probabilities": json.dumps(
                    dict(zip(emotion_labels, predictions.tolist()[0])))  # Convert to JSON
            }

            # Render the results page
            return render_template('results.html',
                                   execution_plan=execution_plan_data)

        except Exception as e:
            # Handle general errors
            return render_template('index.html',
                                  error=f'An error occurred: {e}')

    return render_template('index.html',
                          error='Invalid file type. Allowed types are png, jpeg, jpg.')


if __name__ == '__main__':
    app.run(debug=False)