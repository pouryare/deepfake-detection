import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.getcwd(), 'deepfake_detection.keras')
model = load_model(model_path)

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(os.getcwd(), 'uploads', filename)
        file.save(filepath)
        
        preprocessed_image = preprocess_image(filepath)
        prediction = model.predict(preprocessed_image)
        
        result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
        confidence = float(prediction[0][0]) if result == 'Fake' else float(1 - prediction[0][0])
        
        os.remove(filepath)  # Remove the uploaded file after prediction
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2%}"
        })

if __name__ == "__main__":
    os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)