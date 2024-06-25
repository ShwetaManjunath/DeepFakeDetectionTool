# app.py
import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor
from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Transformation for input frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def extract_frames(video_path, frame_rate=1):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    success, image = video_capture.read()
    while success:
        if frame_count % frame_rate == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            frame = transform(pil_image)
            frames.append(frame)
        success, image = video_capture.read()
        frame_count += 1
    video_capture.release()
    return frames

def predict_frames(frames):
    model_name = "google/vit-base-patch16-224"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    model.eval()
    predictions = []

    for frame in frames:
        # Denormalize frame to original range
        denormalized_frame = denormalize(frame, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        inputs = processor(images=denormalized_frame, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.array(predictions)

def interpret_prediction(avg_prediction, threshold=0.5):
    if len(avg_prediction) == 2:
        fake_prob = avg_prediction[0][1]  # Assuming avg_prediction is a list of arrays with softmax outputs
    else:
        # Assuming avg_prediction is a single array with softmax outputs
        fake_prob = avg_prediction[0][1]  # Change the index [1] if the fake class index is different

    return "Fake video" if fake_prob >= threshold else "Real video"


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
         

        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        frames = extract_frames(video_path, frame_rate=30)  # Extract 1 frame per second
        if not frames:
            return jsonify({"error": "No frames extracted from video"}), 400

        predictions = predict_frames(frames)
        if predictions.size == 0:
            return jsonify({"error": "No predictions made"}), 400

        avg_prediction = np.mean(predictions, axis=0)
        avg_prediction_list = avg_prediction.tolist()
        interpretation = interpret_prediction(avg_prediction_list)

        response_data = {'prediction': avg_prediction_list, 'interpretation': interpretation}

        json_filename = os.path.splitext(file.filename)[0] + '_result.json'
        json_filepath = os.path.join(UPLOAD_FOLDER, json_filename)
        with open(json_filepath, 'w') as json_file:
            json.dump(response_data, json_file)

        return jsonify(response_data)
    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
