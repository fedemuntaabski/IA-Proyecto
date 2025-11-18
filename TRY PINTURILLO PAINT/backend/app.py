from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2
import mediapipe as mp
import threading
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load the model
import os
model_path = os.path.join(os.path.dirname(__file__), 'sketch_classifier_model.keras')
model = tf.keras.models.load_model(model_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variables for hand tracking
current_hand_position = None
is_hand_open = False
camera_active = False
cap = None

# Class names from the trained model (228 classes)
CLASS_NAMES = [
    "The Eiffel Tower", "The Great Wall of China", "The Mona Lisa", "aircraft carrier", "airplane",
    "alarm clock", "ambulance", "arm", "asparagus", "axe", "baseball", "baseball bat", "basket",
    "basketball", "bat", "bathtub", "beach", "bear", "beard", "bed", "bee", "belt", "bench",
    "bicycle", "binoculars", "bird", "birthday cake", "book", "boomerang", "bowtie", "bracelet",
    "brain", "bread", "bridge", "broccoli", "bucket", "bulldozer", "bus", "bush", "butterfly",
    "cactus", "cake", "camel", "camera", "cannon", "canoe", "carrot", "cat", "ceiling fan",
    "cell phone", "chair", "chandelier", "church", "computer", "cookie", "couch", "crab", "crown",
    "dolphin", "donut", "drill", "drums", "duck", "dumbbell", "ear", "elephant", "envelope",
    "eraser", "fan", "feather", "fence", "finger", "firetruck", "fish", "flamingo", "flashlight",
    "flip flops", "floor lamp", "flower", "foot", "fork", "frog", "frying pan", "garden",
    "giraffe", "golf club", "grapes", "grass", "guitar", "hamburger", "hammer", "hand", "harp",
    "hat", "headphones", "helicopter", "helmet", "hexagon", "hot air balloon", "hot dog", "house",
    "jacket", "jail", "knee", "knife", "ladder", "lantern", "laptop", "leaf", "leg", "light bulb",
    "lightning", "lion", "lipstick", "lobster", "lollipop", "map", "matches", "megaphone",
    "mermaid", "microphone", "microwave", "monkey", "moon", "mosquito", "motorbike", "mountain",
    "mouse", "moustache", "mouth", "mug", "mushroom", "nail", "necklace", "nose", "octopus",
    "oven", "panda", "pants", "paper clip", "parachute", "parrot", "passport", "peanut", "peas",
    "pencil", "penguin", "piano", "pickup truck", "pig", "pineapple", "pizza", "pool", "popsicle",
    "power outlet", "purse", "rabbit", "raccoon", "radio", "rain", "rainbow", "remote control",
    "rhinoceros", "rifle", "river", "roller coaster", "rollerskates", "sandwich", "saxophone",
    "scissors", "scorpion", "shark", "sheep", "shoe", "shorts", "shovel", "skateboard", "skull",
    "skyscraper", "smiley face", "snail", "snake", "soccer ball", "sock", "spider", "spoon",
    "squirrel", "stairs", "star", "stethoscope", "stop sign", "stove", "strawberry", "submarine",
    "suitcase", "sun", "swan", "sword", "syringe", "t-shirt", "table", "teapot", "teddy-bear",
    "telephone", "television", "tennis racquet", "tent", "tiger", "toaster", "toothbrush",
    "toothpaste", "tornado", "tractor", "traffic light", "train", "tree", "truck", "trumpet",
    "umbrella", "violin", "washing machine", "whale", "wheel", "windmill", "wine bottle",
    "wine glass", "wristwatch", "zebra"
]

print(f"Model loaded successfully!")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Number of classes: {len(CLASS_NAMES)}")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        
        # Remove the data:image/png;base64, prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        
        # Resize to model's expected input size (usually 28x28 for sketch models)
        input_shape = model.input_shape[1:3]  # Get height and width
        image = image.resize((input_shape[1], input_shape[0]), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        
        # Invert the image (white background with black strokes -> black background with white strokes)
        # This matches the Quick Draw dataset format
        image_array = 255 - image_array
        
        # Normalize to 0-1 range
        image_array = image_array / 255.0
        
        # Reshape for model input
        image_array = image_array.reshape(1, input_shape[0], input_shape[1], 1)
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        results = []
        
        for idx in top_indices:
            class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
            confidence = float(predictions[0][idx])
            percentage = confidence * 100
            results.append({
                'class': class_name,
                'confidence': confidence,
                'percentage': percentage
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'classes': CLASS_NAMES[:model.output_shape[1]]
    })

def check_hand_open(landmarks):
    """Check if hand is open based on thumb and index finger distance"""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )
    
    return distance > 0.1

def process_camera_frame():
    """Process camera frames with hand tracking"""
    global current_hand_position, is_hand_open, cap, camera_active
    
    while camera_active:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        hand_data = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get palm center (landmark 9)
                palm_center = hand_landmarks.landmark[9]
                
                # Convert to canvas coordinates (600x600)
                x = palm_center.x * 600
                y = palm_center.y * 600
                
                is_open = check_hand_open(hand_landmarks.landmark)
                
                hand_data = {
                    'x': x,
                    'y': y,
                    'isOpen': is_open
                }
                
                current_hand_position = hand_data
                is_hand_open = is_open
                
                # Draw hand landmarks on frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Emit frame and hand data via WebSocket
        socketio.emit('video_frame', {
            'frame': base64.b64encode(frame_bytes).decode('utf-8'),
            'hand': hand_data
        })
        
        time.sleep(0.016)  # ~60 FPS

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_camera')
def handle_start_camera():
    global cap, camera_active
    
    if not camera_active:
        camera_active = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=process_camera_frame, daemon=True)
        thread.start()
        
        emit('camera_started', {'success': True})
    else:
        emit('camera_started', {'success': True, 'already_running': True})

@socketio.on('stop_camera')
def handle_stop_camera():
    global cap, camera_active
    
    camera_active = False
    if cap is not None:
        cap.release()
        cap = None
    
    emit('camera_stopped', {'success': True})

@socketio.on('get_hand_position')
def handle_get_hand_position():
    global current_hand_position, is_hand_open
    
    if current_hand_position:
        emit('hand_position', {
            'position': current_hand_position,
            'isOpen': is_hand_open
        })

if __name__ == '__main__':
    print("Starting Flask server with SocketIO...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
