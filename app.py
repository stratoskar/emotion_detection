from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2
import base64  # For encoding images

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the upload folder if it doesn't exist

# Load your trained model
model_path = 'models/emotion_recognition_model_fer_best.h5'  # Adjust path if needed
try:
    model = load_model(model_path)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad',
                      'surprise']
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


def preprocess_image_for_display(img_path):
    """
    Loads and preprocesses the image for display in the browser.
    Returns the original and preprocessed images as base64 encoded strings.
    """
    try:
        # Load the original image
        original_img = cv2.imread(img_path)
        if original_img is None:
            raise ValueError(f"Could not read image: {img_path}")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Preprocess for the model (same as before)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (img_width, img_height),
                                  interpolation=cv2.INTER_AREA)
        img_array = np.expand_dims(img_resized, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0

        # Encode images to base64 for display
        _, original_img_encoded = cv2.imencode('.jpg', original_img)
        original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')

        _, processed_img_encoded = cv2.imencode('.jpg', img_resized)
        processed_img_base64 = base64.b64encode(processed_img_encoded).decode('utf-8')

        return original_img_base64, processed_img_base64, img_array  # Return original, processed, model input

    except Exception as e:
        print(f"Error preprocessing image for display: {e}")
        return None, None, None


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

            # Preprocess the image for display AND model input
            original_img_base64, processed_img_base64, processed_image = preprocess_image_for_display(
                filepath)

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

            # Prepare data for the results page
            results_data = {
                "original_image_base64": original_img_base64,  # Base64 encoded original image
                "processed_image_base64": processed_img_base64,  # Base64 encoded processed image
                "predicted_emotion": predicted_emotion,
                "all_emotions_probabilities": {
                    label: round(float(prob), 3)
                    for label, prob in zip(emotion_labels, predictions.tolist()[0])
                }
            }

            # Render the results page
            return render_template('results.html', results_data=results_data)

        except Exception as e:
            # Handle general errors
            return render_template('index.html',
                                  error=f'An error occurred: {e}')

    return render_template('index.html',
                          error='Invalid file type. Allowed types are png, jpeg, jpg.')


if __name__ == '__main__':
    app.run(debug=False)