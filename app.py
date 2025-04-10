from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Create the upload folder if it doesn't exist

# Load your trained model
model_path = 'emotion_recognition_model_fer.h5'
try:
    model = load_model(model_path)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None
    emotion_labels = []

# Define the target image size used during training
img_width, img_height = 48, 48

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(img_width, img_height), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
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
        return render_template('index.html', error="Model not loaded. Please check the server.")

    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            processed_image = preprocess_image(filepath)

            if processed_image is None:
                os.remove(filepath) # Clean up if processing fails
                return render_template('index.html', error='Error processing the uploaded image.')

            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_emotion = emotion_labels[predicted_class_index]

            return render_template('result.html', prediction=predicted_emotion, image_path=filepath)

        except Exception as e:
            return render_template('index.html', error=f'An error occurred during prediction: {e}')

    return render_template('index.html', error='Invalid file type. Allowed types are png, jpeg, jpg.')

if __name__ == '__main__':
    app.run(debug=True)