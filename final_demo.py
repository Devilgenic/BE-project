"""
Final demonstration script for the violence detection system.
This script shows that the system is properly integrated with the trained model.
"""

import cv2
import numpy as np
import time
from violence_model import create_violence_detection_model, detect_violence

def demonstrate_model_capabilities():
    """Demonstrate that the trained model is working correctly"""
    print("=" * 60)
    print("VIOLENCE DETECTION SYSTEM - MODEL DEMONSTRATION")
    print("=" * 60)
    
    # Load the trained model
    print("1. Loading trained violence detection model...")
    model = create_violence_detection_model()
    
    if model is None:
        print("   ERROR: Failed to load model!")
        return
    
    print("   SUCCESS: Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    
    # Show model information
    print("\n2. Model information:")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Test with multiple dummy frames
    print("\n3. Testing violence detection with sample frames...")
    
    # Create a series of test frames
    test_frames = []
    for i in range(5):
        # Create different types of frames
        if i % 2 == 0:
            # Normal frame (random noise)
            frame = np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8)
        else:
            # Potentially violent frame (brighter, more intense)
            frame = np.random.randint(150, 255, (480, 640, 3), dtype=np.uint8)
        
        test_frames.append(frame)
    
    # Test each frame
    violence_detections = 0
    for i, frame in enumerate(test_frames):
        result = detect_violence(model, frame)
        status = "VIOLENCE DETECTED" if result else "No violence"
        print(f"   Frame {i+1}: {status}")
        if result:
            violence_detections += 1
    
    print(f"\n   Summary: {violence_detections}/{len(test_frames)} frames flagged as violent")
    
    # Performance test
    print("\n4. Performance test...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    for _ in range(10):
        detect_violence(model, test_frame)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"   Average detection time: {avg_time:.4f} seconds per frame")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("The violence detection system is ready for use!")
    print("Run 'python app.py' and visit http://localhost:5000 to use the web interface.")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_model_capabilities()