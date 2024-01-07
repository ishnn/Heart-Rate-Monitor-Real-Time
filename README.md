# Heart Rate Measurement System with Jarvis Interaction

## Introduction

This Python script utilizes computer vision and signal processing techniques to measure a person's heart rate in real-time using a webcam. The heart rate is estimated by analyzing facial landmarks and extracting color information from the user's face. The system is integrated with Jarvis, a virtual assistant, to provide a voice-based user interface for initiating heart rate measurements and receiving results.

## Requirements

Ensure you have the following libraries installed:
dlib==19.24.1
imutils==0.5.4
matplotlib==3.8.2
numpy==1.23.5
opencv_contrib_python==4.6.0.66
opencv_contrib_python_headless==4.8.1.78
opencv_python==4.6.0.66
pyttsx3==2.90
scipy==1.11.4
SpeechRecognition==3.10.0


## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/ishnn/Heart-Rate-Monitor-Real-Time
   cd Heart-Rate-Monitor-Real-Time
   ```

2. Download the Dlib shape predictor model file from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it into the project directory.

3. Run the script:

   ```bash
   python heart_rate_measurement.py
   ```

## Usage

1. When the script is running, say "Jarvis, check my heart rate" to initiate the heart rate measurement process.

2. A window will open showing your face with detected facial landmarks. The estimated heart rate (in beats per minute) will be displayed on the window.

3. After the measurement is complete, Jarvis will announce the average heart rate.

## Notes

- The script captures facial landmarks and uses color information from specific regions on the face to estimate the heart rate.

- Ensure good lighting conditions for accurate results.

- The heart rate measurement is performed in the background while the user interacts with Jarvis.

- The heart rate results are stored in a CSV file (`ppg_signal.csv`)
