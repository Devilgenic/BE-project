import cv2
import numpy as np
from violence_model import create_violence_detection_model, detect_violence

def test_model_loading():
    """Test that the violence detection model loads correctly"""
    print("Testing model loading...")
    
    # Load the model
    model = create_violence_detection_model()
    
    if model is None:
        print("ERROR: Failed to load model")
        return False
    else:
        print(f"SUCCESS: Model loaded successfully")
        print(f"Model type: {type(model)}")
        return True

def test_violence_detection():
    """Test violence detection with a dummy frame"""
    print("\nTesting violence detection...")
    
    # Load the model
    model = create_violence_detection_model()
    
    if model is None:
        print("ERROR: Cannot test detection without model")
        return False
    
    # Create a dummy frame (random noise)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test violence detection
    try:
        is_violent = detect_violence(model, dummy_frame)
        print(f"Detection result: {'VIOLENCE DETECTED' if is_violent else 'No violence'}")
        print("SUCCESS: Violence detection function worked")
        return True
    except Exception as e:
        print(f"ERROR: Violence detection failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Running model tests...\n")
    
    # Test model loading
    load_success = test_model_loading()
    
    # Test violence detection
    detection_success = False
    if load_success:
        detection_success = test_violence_detection()
    
    print("\n" + "="*50)
    if load_success and detection_success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*50)