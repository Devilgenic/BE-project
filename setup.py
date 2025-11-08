"""
Setup script for Violence Detection System
"""
import os
import sys

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['static', 'templates', 'uploads']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'cv2', 'numpy', 'requests']
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print("  pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main setup function"""
    print("Setting up Violence Detection System...")
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\nSetup completed successfully!")
    print("\nTo run the application:")
    print("  python app.py")
    print("\nThen open your browser and go to http://localhost:5000")

if __name__ == "__main__":
    main()