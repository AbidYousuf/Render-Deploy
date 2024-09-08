from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the Haar Cascade for face detection and the emotion classifier model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('modelA.keras')

# Emotion labels that the model predicts
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to process and predict the emotion from the image
def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_color = image[y:y + h, x:x + w]
        roi_color_resized = cv2.resize(roi_color, (224, 224), interpolation=cv2.INTER_AREA)

        roi = img_to_array(roi_color_resized)
        roi = np.expand_dims(roi, axis=0)
        roi = roi.astype('float') / 255.0

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        return label
    return "No Face Detected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    label = predict_emotion(image)

    return jsonify({'emotion': label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
