from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from io import BytesIO
from PIL import Image
import pickle
import os
import json
from typing import List
# Initialize FastAPI
app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend domain if restricted
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_converted_val_loss_0.0865_224x224(17-1-24).tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9)

def preprocess_frame(frame, bbox):
    """Preprocess the frame for model input."""
    x_min, y_min, x_max, y_max = bbox

    # Ensure bounding box coordinates are within frame dimensions
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)

    if x_min >= x_max or y_min >= y_max:
        return None

    # Crop face region
    face = frame[y_min:y_max, x_min:x_max]

    # Resize to target size (224x224)
    resized_face = cv2.resize(face, (224, 224))

    # Normalize the image to [0, 1] range
    normalized_face = resized_face / 255.0

    # Convert to (Batch, Height, Width, Channels)
    processed_face = np.expand_dims(normalized_face, axis=0)

    return processed_face.astype(np.float32)

# Pastikan fungsi preprocess_regist didefinisikan dengan benar
def preprocess_regist(image, bbox):
    # Misalnya, Anda bisa mengubah ukuran wajah atau menerapkan preprocessing lainnya
    x1, y1, x2, y2 = bbox
    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face_resized = cv2.resize(face, (224, 224))
    return [face_resized]  # Return as list

def save_to_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

@app.post("/register")
async def register(
    frame: UploadFile = File(...),
    bbox: str = Form(...),
    username: str = Form(...),
):
    try:
        bbox = json.loads(bbox)  # Parse the bbox as JSON string
    except json.JSONDecodeError as e:
        return JSONResponse({"message": f"Invalid bounding box format: {e}"}, status_code=400)

    if not bbox or not username:
        return JSONResponse({"message": "No bounding box or username provided."}, status_code=400)

    # Read uploaded image
    image_data = await frame.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Check if the image is loaded correctly
    if image is None:
        return JSONResponse({"message": "Failed to read the image."}, status_code=400)

    # Log image dimensions for debugging
    print(f"Image dimensions: {image.shape}")

    # Process frame and save image
    processed_face = preprocess_regist(image, bbox)
    print(processed_face)
    if processed_face is None:
        return JSONResponse({"message": "Invalid face detected."}, status_code=400)

    # Create directories per user
    base_dir = "base_directory"  # Ganti dengan direktori dasar Anda
    images_dir = os.path.join(base_dir, "captured_images", username)
    model_dir = os.path.join(base_dir, "model", username)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save image to user's directory
    image_count = len([img for img in os.listdir(images_dir) if img.startswith(username)])
    image_path = os.path.join(images_dir, f"{username}_{image_count + 1}.png")
    
    # Log the image being saved for debugging
    print(f"Saving image to {image_path}")
    
    # Check if the processed face is valid
    if processed_face[0] is None:
        return JSONResponse({"message": "Processed face is None."}, status_code=400)

    cv2.imwrite(image_path, processed_face[0])

    # Save to pickle
    model_images_path = os.path.join(model_dir, f"{username}_images.pkl")
    if os.path.exists(model_images_path):
        with open(model_images_path, 'rb') as f:
            user_images = pickle.load(f)
    else:
        user_images = []


    user_images.append(processed_face[0].flatten())
    save_to_pickle(user_images, model_images_path)

    model_labels_path = os.path.join(model_dir, f"{username}_labels.pkl")
    labels = [username] * len(user_images)
    save_to_pickle(labels, model_labels_path)

    return {"message": f"Registration successful for user '{username}'."}

@app.post("/predict")
async def predict(frame: UploadFile = File(...)):
    import cv2
    import numpy as np

    # Read uploaded image
    image_data = await frame.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Convert to RGB for Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face using Mediapipe
    results = face_detection.process(rgb_image)
    if not results.detections:
        return JSONResponse({"label": "No face detected", "bbox": None})

    # Assume the first detected face (you can extend to multiple faces if needed)
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    h, w, _ = image.shape
    x1 = int(bboxC.xmin * w)
    y1 = int(bboxC.ymin * h)
    x2 = int((bboxC.xmin + bboxC.width) * w)
    y2 = int((bboxC.ymin + bboxC.height) * h)

    # Expand bounding box slightly
    expansion_factor_width = 0.3
    expansion_factor_height = 0.2
    width_margin = int(expansion_factor_width * (x2 - x1))
    height_margin = int(expansion_factor_height * (y2 - y1))
    x1 = max(0, x1 - width_margin)
    y1 = max(0, y1 - (height_margin + 80))
    x2 = min(w, x2 + width_margin)
    y2 = min(h, y2 + height_margin)

    # Preprocess face for TFLite model
    processed_face = preprocess_frame(image, (x1, y1, x2, y2))
    if processed_face is None:
        return JSONResponse({"label": "Invalid face", "bbox": None})

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], processed_face)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_prob = predictions[0][0]
    # print(predicted_prob)
    # print(x1,y1,x2,y2)
    # Return result with bounding box
    label = "Real" if predicted_prob > 0.7 else "Fake"
    return {
        "label": label,
        "probability": float(predicted_prob),
        "bbox": [int(x1), int(y1), int(x2), int(y2)]
    }

