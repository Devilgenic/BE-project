# Violence Detection System

A web-based violence detection system using deep learning that can analyze uploaded videos or real-time webcam feeds to detect violent activities. When violence is detected, the system triggers an audible alarm and sends an alert with the captured frame to a Telegram bot.

## Features

- Upload video files for violence detection
- Real-time webcam violence detection
- Audible alarm when violence is detected
- Telegram bot integration for alerts
- Web-based user interface

## Requirements

- Python 3.7+
- Flask
- OpenCV
- TensorFlow
- NumPy
- Requests

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   
   Or run the setup script:
   ```
   python setup.py
   ```

## Setup Telegram Bot

1. Create a Telegram bot using BotFather:
   - Open Telegram and search for @BotFather
   - Send `/newbot` and follow the instructions to create a new bot
   - Copy the bot token provided

2. Get your chat ID:
   - Search for your newly created bot in Telegram
   - Send any message to the bot
   - Visit this URL in your browser (replace `YOUR_BOT_TOKEN` with your actual token):
     `https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates`
   - Look for the "id" field in the response - this is your chat ID

3. Update the configuration in `app.py`:
   ```python
   TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
   TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and go to `http://localhost:5000`

3. You can either:
   - Upload a video file for analysis
   - Start real-time webcam detection

4. When violence is detected:
   - An audible alarm will sound
   - The frame where violence was detected will be sent to your Telegram bot

## How It Works

The system uses a deep learning model to analyze video frames for signs of violence. The system now uses your trained model [modelnew.h5](file:///C:/Users/prash/OneDrive/Desktop/try/modelnew.h5) for violence detection.

The detection process:
1. Video frames are captured (from uploaded video or webcam)
2. Frames are preprocessed and fed to the trained violence detection model
3. If violence is detected:
   - An audible alarm is triggered
   - The frame is saved as an image
   - The image is sent to the configured Telegram bot

## Customization

The system is now using your trained violence detection model [modelnew.h5](file:///C:/Users/prash/OneDrive/Desktop/try/modelnew.h5). To further customize:

1. If you need to change the preprocessing steps, modify the [preprocess_frame()](file:///C:/Users/prash/OneDrive/Desktop/try/violence_model.py#L14-L26) function in `violence_model.py`
2. If your model has different output format, update the [detect_violence()](file:///C:/Users/prash/OneDrive/Desktop/try/violence_model.py#L28-L52) function in `violence_model.py`
3. Adjust the detection threshold if needed

## Limitations

This is a demonstration implementation. For production use, consider:
- Implementing additional security measures
- Adding more robust error handling
- Optimizing performance for real-time processing
- Fine-tuning the model for better accuracy