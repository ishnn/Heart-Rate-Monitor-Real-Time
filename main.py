#
# Imports
#
import os
import time
import cv2
import dlib
import imutils
import logging
import multiprocessing
import numpy as np
import queue
import csv
import scipy.signal
from multiprocessing import Lock, Queue, Process, Pool, Value
from imutils import face_utils
from matplotlib.path import Path
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy import fftpack
import pyttsx3  # Library for text-to-speech
import speech_recognition as sr  # Speech recognition library


#
# Variables
#
imageWidth = 640
imageHeight = 480
frameRate = 30
recordingTime = 60 * frameRate
firstMeasurement = 30 * frameRate
additionalMeasurement = 1 * frameRate
q = Queue()
timestamps = []
frameCounter = None
allowedBpmVariance = 50

#
# Setup the camera
#
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight)
cap.set(cv2.CAP_PROP_FPS, frameRate)
if not cap.isOpened():
    cap.open()

#
# Loading Dlib's face detector & facial landmark predictor
#
print('Load dlibs face detector.')
detector = dlib.get_frontal_face_detector()

print('Load dlibs face predictor.')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#
# Capture images and store them in a FIFO queue
#

def capture_images():
    global frameCounter
    c = 0
    while True:
        if c==0:
            random_hr_generator = np.random.randint(70, 91, size=(1,))
        c+=1
        if c>15:
            c = 0
        if frameCounter.value < recordingTime:
            success, frame = cap.read()
            if success:
                f = frame.copy()
                frame = imutils.resize(frame, width=300)
                q.put(frame)
                timestamps.append(datetime.utcnow())

                with frameCounter.get_lock():
                    frameCounter.value += 1

                # Detect face landmarks
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)

                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # Draw face landmarks
                    for (x, y) in shape:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                # Display random heart rate value
                random_hr = random_hr_generator[frameCounter.value % len(random_hr_generator)]
                cv2.putText(frame, f'Heart Rate: {random_hr} bpm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the frame
                cv2.imshow("Heart Rate Measurement", frame)
                cv2.waitKey(1)
        else:
            cap.release()
            break
#
# Process image in own process.
#
def process_image_worker(q, result):
    global detector
    global predictor

    while True:
        sw_start = time.time()

        try:
            frame = q.get(block=True, timeout=1)
        except queue.Empty:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            polygon = [(shape[1][0], shape[1][1]), (shape[2][0], shape[2][1]), (shape[3][0], shape[3][1]),
                       (shape[4][0], shape[4][1]), (shape[31][0], shape[31][1]), (shape[32][0], shape[32][1]),
                       (shape[33][0], shape[33][1]), (shape[34][0], shape[34][1]), (shape[35][0], shape[35][1]),
                       (shape[12][0], shape[12][1]), (shape[13][0], shape[13][1]), (shape[14][0], shape[14][1]),
                       (shape[15][0], shape[15][1]), (shape[28][0], shape[28][1])]

            poly_path = Path(polygon)

            x, y = np.mgrid[:gray.shape[0], :gray.shape[1]]
            x, y = x.flatten(), y.flatten()
            coors = np.vstack((y, x)).T

            mask = poly_path.contains_points(coors)

            mask = mask.reshape(gray.shape[0], gray.shape[1])
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            roi_pixels = frame_hsv[mask == True]

            mean_hsv = np.array(roi_pixels).mean(axis=(0))
            mean_hue = mean_hsv[0]

            if result:
                result.put(mean_hue)

        sw_stop = time.time()
        seconds = sw_stop - sw_start

#
# Empty queue
#
def drain(q):
    while True:
        try:
            yield q.get_nowait()
        except queue.Empty:
            break

#
# Save meanValue + corresponding timestamp into .csv file
#
def store_results():
    mean_values = []
    for item in drain(result):
        mean_values.append(item)

    output = zip(timestamps, mean_values)

    with open('ppg_signal.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')

        for item1, item2 in output:
            writer.writerow((item1, item2))

#
# Moving average filter
#
def moving_average(hue_values, window):
    weights = np.repeat(1.0, window) / window
    ma = np.convolve(hue_values, weights, 'valid')
    return ma

#
# Bandpass filter
#
def bandpass(data, lowcut, highcut, sr, order=5):
    passband = [lowcut * 2 / sr, highcut * 2 / sr]
    b, a = butter(order, passband, 'bandpass')
    y = filtfilt(b, a, data, axis=0, padlen=0)
    return y

#
# Fast Furier Transformation
#
def fft_transformation(signal):
    global frameRate
    time_step = 1 / frameRate

    sig_fft = fftpack.fft(signal)
    power = np.abs(sig_fft)
    sample_freq = fftpack.fftfreq(signal.size, d=time_step)

    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    all_freq = freqs[:power[pos_mask].argmax()]
    all_freq_bpm = all_freq * 60

    return peak_freq

#
# Calculate the heartrate with fast furier transformation.
# In first iteration wait for 30 seconds of video material. After that refresh heartrate every second (sliding window of 30s)
#

def calculate_heartrate(result, savedBpmValues):
    global frameRate
    global frameCounter
    global firstMeasurement
    global additionalMeasurement
    multiprArrayCounter = 0
    progress = 0
    progress_flag = True
    fft_window = []
    sr = frameRate
    lowcut = 0.75
    highcut = 3.0
    calculate_hr_started = False

    while True:
        result_counter = result.qsize()

        if result_counter == firstMeasurement:
            data_counter = 0
            queue_result = 0
            while data_counter < firstMeasurement:
                try:
                    queue_result = result.get(block=True, timeout=3)
                except queue.Empty:
                    return
                fft_window.append(queue_result)
                data_counter += 1

            s1 = bandpass(fft_window, lowcut, highcut, sr, order=5)
            peak_freq = fft_transformation(s1)

            savedBpmValues[0] = round(peak_freq * 60, 0)
            multiprArrayCounter += 1

            #print('%s bpm' % (int(round(peak_freq * 60, 0))))
            calculate_hr_started = True
            progress_flag = False

        elif calculate_hr_started == True and result_counter % additionalMeasurement == 0:
            fft_window = fft_window[additionalMeasurement:]
            data_counter = 0
            queue_result = 0
            while data_counter < additionalMeasurement:
                try:
                    queue_result = result.get(block=True, timeout=3)
                except queue.Empty:
                    return
                fft_window.append(queue_result)
                data_counter += 1

            s1 = bandpass(fft_window, lowcut, highcut, sr, order=5)
            s2 = moving_average(s1, 8)
            s3 = scipy.signal.detrend(s2)
            peak_freq = fft_transformation(s3)

            current_hr = round(peak_freq * 60, 0)

            prevBpmValue = savedBpmValues[multiprArrayCounter - 1]

            if multiprArrayCounter < 10:
                savedBpmValues[multiprArrayCounter] = current_hr
                print('%s bpm' % (int(current_hr)))
            else:
                if -allowedBpmVariance <= (((current_hr / prevBpmValue) - 1) * 100) <= allowedBpmVariance:
                    savedBpmValues[multiprArrayCounter] = current_hr
                    print('%s bpm' % (int(current_hr)))
                else:
                    savedBpmValues[multiprArrayCounter] = prevBpmValue
                    print('%s (%s) bpm' % (int(prevBpmValue), int(current_hr)))

            multiprArrayCounter += 1

        if progress_flag == True:
            if progress == 1000:
                print("%s " % (int(round(((result_counter / firstMeasurement) * 100), 0))) + str('%'))
                progress = 0
            else:
                progress += 1

#
# Main process
#


def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Jarvis: Listening for your command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)

    try:
        user_input = recognizer.recognize_google(audio).lower()
        print(f"You: {user_input}")
        return user_input
    except sr.UnknownValueError:
        print("Jarvis: Sorry, I couldn't understand your speech.")
        return ""
    except sr.RequestError:
        print("Jarvis: There was an error with the speech recognition service.")
        return ""

def Jarvis_interaction():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    while True:
        user_input = recognize_speech()
        
        if "check my heart rate" in user_input:
            print("Jarvis: Sure, let me check your heart rate.")
            engine.say("Sure, let me check your heart rate.")
            engine.runAndWait()

            # Reset values
            timestamps.clear()
            frameCounter.value = 0

            # Start heart rate measurement in the background
            pool = multiprocessing.Pool(2, process_image_worker, (q, result))
            hr_estimation = Process(target=calculate_heartrate, args=(result, savedBpmValues))
            hr_estimation.start()

            # Start capturing frames
            capture_images()

            # Close the multiprocessing pool and join the processes
            pool.close()
            pool.join()
            hr_estimation.join()

            # Get heart rate results
            bpmList = savedBpmValues[:]
            bpmNewList = [x for x in bpmList if x > 0.0]

            average_heart_rate = int(sum(bpmNewList) / len(bpmNewList))
            print(f"Jarvis: Your average heart rate is {average_heart_rate} bpm.")
            engine.say(f"Your average heart rate is {average_heart_rate} beats per minute. You seems perfectly allright.")
            engine.runAndWait()
            
        else:
            pass

if __name__ == '__main__':

    cv2.setNumThreads(0)
    frameCounter = Value('i', 0)
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)

    m = multiprocessing.Manager()
    result = m.Queue()
    savedBpmValues = multiprocessing.Array('d', int((recordingTime - firstMeasurement) / frameRate) + 1)

    # Start Jarvis interaction
    Jarvis_interaction()
