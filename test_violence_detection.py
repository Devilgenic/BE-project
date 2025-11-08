"""
Test script for Violence Detection System
"""
import cv2
import numpy as np
from violence_model import create_violence_detection_model, detect_violence

def create_test_frame():
    """Create a simple test frame for testing"""
    # Create a black image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some random shapes to make it more interesting
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(frame, (400, 200), 50, (0, 255, 0), -1)
    
    return frame

def test_violence_detection():
    """Test the violence detection functionality"""
    print("Testing Violence Detection System...")
    
    # Create the model
    model = create_violence_detection_model()
    print("Model created successfully")
    
    # Create a test frame
    test_frame = create_test_frame()
    print("Test frame created")
    
    # Test violence detection
    is_violent = detect_violence(model, test_frame)
    print(f"Violence detected: {is_violent}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_violence_detection()