# Violence Detection System

An AI-powered real-time violence detection system using **Multi-Signal Temporal Fusion** — combining a Convolutional Neural Network (CNN) with Optical Flow analysis and Motion Energy computation for robust detection in live video feeds. Built with Flask, TensorFlow, and OpenCV.

## Features

- **Multi-Signal Detection Engine** — CNN + Optical Flow + Motion Energy fusion with sliding window evidence accumulation
- **Real-Time Dashboard** — Live video feed, confidence gauge, signal breakdown bars, recent alerts
- **Multiple Video Sources** — Webcam, RTSP/IP camera, and video file upload
- **Detection History** — SQLite-backed event log with timestamps, scores, and captured frames
- **Telegram Alerts** — Automatic photo notifications via Telegram Bot API
- **Configurable Settings** — Sensitivity, signal weights, frame skip, alert cooldown, camera source
- **Architecture Documentation** — Built-in SVG architecture diagram and algorithm explanation page
- **Thread-Safe Design** — Proper locking, daemon threads, structured logging
- **Responsive UI** — Sidebar navigation, mobile-friendly layout

## System Architecture

```
[Video Source] --> [Frame Capture] --> [Preprocessing]
                                            |
                          +-----------------+-----------------+
                          |                 |                 |
                    [CNN Model]      [Optical Flow]    [Motion Energy]
                      (50%)             (30%)              (20%)
                          |                 |                 |
                          +-----------------+-----------------+
                                            |
                               [Temporal Fusion Engine]
                               - Weighted combination
                               - EMA smoothing
                               - Sliding window (30 frames)
                               - Evidence accumulation
                                            |
                               [Decision Logic]
                               - Suspicious ratio > 40%
                               - Recent peak > threshold
                               - Smoothed score check
                                            |
                         +------------------+------------------+
                         |                  |                  |
                    [Alarm]           [Telegram]          [Database]
```

## Algorithm Details

### Multi-Signal Temporal Fusion

The system fuses three complementary signals:

| Signal | Weight | Method | Violence Signature |
|--------|--------|--------|--------------------|
| CNN | 50% | TensorFlow/Keras model (128x128 input) | Spatial patterns of violence |
| Optical Flow | 30% | Farneback dense optical flow | High magnitude + high variance (chaotic motion) |
| Motion Energy | 20% | Frame differencing + thresholding | Large-area sudden pixel changes |

### Evidence Accumulation

Violence is confirmed only when **all three conditions** are met over a sliding window of 30 frames:
1. **Suspicious Ratio > 40%** — More than 40% of window frames have fusion score > 0.35
2. **Recent Peak > Threshold** — At least one of the last 5 frames exceeds the confidence threshold
3. **Smoothed Score > 80% of Threshold** — EMA-smoothed score is sufficiently high

This prevents both false positives (single spikes) and false negatives (brief dips).

## Requirements

- Python 3.7+
- Flask
- OpenCV
- TensorFlow
- NumPy
- Requests

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd project

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser at **http://localhost:5000**

## Configuration

Settings can be configured via the **Settings** page in the web UI, or by editing `config.json`:

```json
{
  "telegram": { "bot_token": "", "chat_id": "" },
  "detection": { "sensitivity": "medium", "frame_skip": 3, "alert_cooldown_seconds": 10 },
  "weights": { "cnn_weight": 0.50, "optical_flow_weight": 0.30, "motion_energy_weight": 0.20 },
  "camera": { "source": "webcam", "webcam_index": 0, "rtsp_url": "" }
}
```

### Telegram Bot Setup

1. Create a bot via **@BotFather** on Telegram
2. Copy the bot token
3. Send a message to your bot, then get your chat ID from `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Enter both values in the Settings page

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Flask application — routes, video processing, API endpoints, SSE streaming |
| `detection_engine.py` | ViolenceDetector class — multi-signal fusion, sliding window, EMA, decision logic |
| `violence_model.py` | CNN model loader — loads modelnew.h5, preprocessing |
| `config.py` | Thread-safe configuration manager with JSON persistence |
| `database.py` | SQLite manager for detection event history |
| `templates/base.html` | Shared layout with sidebar navigation |
| `templates/index.html` | Dashboard — live feed, gauge, signal bars, alerts |
| `templates/history.html` | Detection event log with pagination |
| `templates/settings.html` | Configuration panel |
| `templates/about.html` | Architecture diagram and algorithm explanation |
| `static/style.css` | Complete responsive CSS |
| `modelnew.h5` | Trained CNN violence detection model |

## Technology Stack

- **Python 3.x** — Primary language
- **Flask** — Web framework with REST API
- **TensorFlow/Keras** — Deep learning model inference
- **OpenCV** — Video capture, optical flow, frame differencing
- **SQLite** — Detection event database
- **Telegram Bot API** — Alert notifications
- **HTML5/CSS3/JavaScript** — Responsive web interface
- **Server-Sent Events (SSE)** — Real-time confidence streaming
