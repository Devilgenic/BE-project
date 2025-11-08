import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
import requests
import threading
import time
from violence_model import create_violence_detection_model, detect_violence

app = Flask(__name__)
app.static_folder = 'static'

# Global variables for violence detection
violence_detected = False
current_frame = None
detection_active = False
violence_frame_count = 0  # Counter for consecutive violence detections
VIOLENCE_THRESHOLD = 2  # Number of consecutive frames needed to confirm violence (reduced from 3 to 2)

# Load pre-trained violence detection model
model = create_violence_detection_model()

# Telegram bot configuration
TELEGRAM_BOT_TOKEN = "8326099202:AAEUivJ5mfBFHCerZYqzVh8dB7-XnsWsa8Y"
TELEGRAM_CHAT_ID = "913931476"

def send_telegram_alert(image_path):
    """Send alert with image to Telegram bot"""
    try:
        if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
            print("Telegram bot token not configured. Skipping alert.")
            return
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': 'VIOLENCE DETECTED!'}
            response = requests.post(url, files=files, data=data)
            print(f"Telegram response: {response.status_code}")
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")

def detect_violence_in_frame(frame):
    """Detect violence in a frame using the violence detection model"""
    global model
    return detect_violence(model, frame)

def process_video(video_path):
    """Process uploaded video for violence detection"""
    global violence_detected, current_frame, detection_active, violence_frame_count
    
    detection_active = True
    violence_detected = False
    violence_frame_count = 0
    
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while cap.isOpened() and detection_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_frame = frame.copy()
        
        # Process every 5th frame to reduce computational load
        if frame_count % 5 == 0:
            if detect_violence_in_frame(frame):
                violence_frame_count += 1
                print(f"🚨 Violence detected in frame {frame_count}. Count: {violence_frame_count}/{VIOLENCE_THRESHOLD}")
                
                # Only trigger alert if violence detected in multiple consecutive checks
                if violence_frame_count >= VIOLENCE_THRESHOLD:
                    violence_detected = True
                    
                    # Save frame with violence detected
                    timestamp = int(time.time())
                    image_path = f"violence_frame_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame)
                    
                    # Send alert
                    send_telegram_alert(image_path)
                    
                    # Play alarm sound (this would be handled on the frontend)
                    print("⚠️ VIOLENCE CONFIRMED! ALARM TRIGGERED!")
                    
                    # For demo purposes, we'll stop after first detection
                    # In real implementation, you might want to continue monitoring
                    break
            else:
                # Gradually decrease counter if no violence detected
                if violence_frame_count > 0:
                    violence_frame_count -= 0.5
                    violence_frame_count = max(0, violence_frame_count)
        
        # Small delay to prevent excessive processing
        time.sleep(0.01)
    
    cap.release()
    detection_active = False

def process_webcam():
    """Process webcam feed for violence detection"""
    global violence_detected, current_frame, detection_active, violence_frame_count
    
    detection_active = True
    violence_detected = False
    violence_frame_count = 0
    
    cap = cv2.VideoCapture(0)  # Use default camera
    
    frame_count = 0
    while detection_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_frame = frame.copy()
        
        # Process every 5th frame for better responsiveness (changed from 10th)
        if frame_count % 5 == 0:
            if detect_violence_in_frame(frame):
                violence_frame_count += 1
                print(f"🚨 Violence detected in webcam frame! Count: {violence_frame_count}/{VIOLENCE_THRESHOLD}")
                
                # Only trigger alert if violence detected in multiple consecutive checks
                if violence_frame_count >= VIOLENCE_THRESHOLD:
                    violence_detected = True
                    
                    # Save frame with violence detected
                    timestamp = int(time.time())
                    image_path = f"violence_frame_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame)
                    
                    # Send alert
                    send_telegram_alert(image_path)
                    
                    # Play alarm sound (this would be handled on the frontend)
                    print("⚠️ VIOLENCE CONFIRMED! ALARM TRIGGERED!")
                    
                    # Reset counter after alert to avoid multiple alerts
                    violence_frame_count = 0
                    time.sleep(3)  # Wait 3 seconds before checking again (reduced from 5)
            else:
                # Gradually decrease counter if no violence detected (slower decay)
                if violence_frame_count > 0:
                    violence_frame_count -= 0.5
                    violence_frame_count = max(0, violence_frame_count)
        
        # Small delay to prevent excessive processing
        time.sleep(0.05)
    
    cap.release()
    detection_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global violence_detected
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Reset violence detection flag for new upload
    violence_detected = False
    
    # Save uploaded video
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    filename = 'uploads/uploaded_video.mp4'
    file.save(filename)
    
    # Process video in background thread
    thread = threading.Thread(target=process_video, args=(filename,))
    thread.start()
    
    return jsonify({'message': 'Video uploaded successfully. Processing started.'})

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global violence_detected
    
    # Reset violence detection flag for new webcam session
    violence_detected = False
    
    # Start webcam processing in background thread
    thread = threading.Thread(target=process_webcam)
    thread.start()
    
    return jsonify({'message': 'Webcam processing started.'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active, violence_detected
    detection_active = False
    violence_detected = False
    return jsonify({'message': 'Detection stopped.'})

@app.route('/reset', methods=['POST'])
def reset_system():
    global detection_active, violence_detected, violence_frame_count
    detection_active = False
    violence_detected = False
    violence_frame_count = 0
    return jsonify({'message': 'System reset successfully.'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route for webcam feed"""
    def generate():
        global current_frame
        while detection_active:
            if current_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 fps
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/status')
def status():
    return jsonify({
        'violence_detected': violence_detected,
        'detection_active': detection_active
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)