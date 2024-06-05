import cv2
import time
import mediapipe as mp

# Supporting functionality
import math
import threading
import numpy as np

# Importing tensorflow model
from keras.models import load_model

# Audio Control
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Mouse Control
import mouse

# Global Vars
width = 640
height = 480
padding = 100
delay = 0
actions = np.array(['decreaseVol', 'increaseVol', 'scrollUp', 'scrollDown'])
sequence = []

def timeDelay():
    global delay
    global delayThread
    time.sleep(1)
    delay = 0
    delayThread = threading.Thread(target=timeDelay)


def main():
    global delay, sequence
    global delayThread
    delayThread = threading.Thread(target=timeDelay)
    fps = 0
    frames = 0
    display_time = 0.5
    start_time = time.time()

    # Initializing Camera
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Loading the Hand Gesture Model
    mpHands = mp.solutions.hands
    handDetector = mpHands.Hands(
        static_image_mode="store_true",
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mpDrawing = mp.solutions.drawing_utils

    # Loading the Hand Action Model
    action = load_model("trainedModel.h5")

    while True:

        # FPS Calculation
        time_diff = time.time() - start_time
        frames += 1
        if time_diff >= display_time:
            fps = frames / time_diff
            print(fps)
            fps = int(fps)
            frames = 0
            start_time = time.time()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        ret, frame = cam.read()

        # Mirror the frame
        image = cv2.flip(frame, 1)

        # Mediapipe Hand Detection
        image, results = handDetection(image, handDetector)

        # Initializing Volume Control
        volume, minVol, maxVol = initVol()


        # If a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Drawing Landmarks
                drawLandmarks(mpDrawing, image, hand_landmarks, mpHands)

                # Extracting key datapoints
                landmarks = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
                # print(landmarks.shape)

                # Finger Recognition
                fingers = fingerRecognition(landmarks)
                # print(fingers)

                # Mouse Movement
                if fingers['4'] == 1 and fingers['8'] == 1 and fingers['12'] == 0 and fingers['16'] == 0 and fingers[
                    '20'] == 0:
                    x, y = getIndexCoords(landmarks[8][0], landmarks[8][1])
                    cv2.circle(image, (int(x), int(y)), 5, (123, 255, 0), 2)
                    cv2.rectangle(image, (padding, padding), (width - padding, height - padding), (255, 0, 0), 2)
                    cx, cy = convertCoords((x, y))
                    mouse.move(cx, cy)

                # Left Click
                if fingers['4'] == 0 and fingers['8'] == 1 and fingers['12'] == 0 and fingers['16'] == 0 and fingers['20'] == 0:
                    ind_x, ind_y = getIndexCoords(landmarks[8][0], landmarks[8][1])
                    thumb_x, thumb_y = getIndexCoords(landmarks[4][0], landmarks[4][1])

                    if abs(ind_x - thumb_x) < 25:
                        if delay == 0:
                            mouse.click(button="left")
                            delay = 1
                            delayThread.start()

                # Volume Control
                if fingers['4'] == 0 and fingers['8'] == 1 and fingers['12'] == 1 and fingers['16'] == 0 and fingers['20'] == 0:
                    ind_x, ind_y = getIndexCoords(landmarks[8][0], landmarks[8][1])
                    thumb_x, thumb_y = getIndexCoords(landmarks[4][0], landmarks[4][1])
                    drawVolume(image, ind_x, ind_y, thumb_x, thumb_y)
                    dist = math.hypot(ind_x - thumb_x, ind_y - thumb_y)

                    # The length depends on distance from the camera
                    vol = np.interp(dist, [30, 100], [minVol, maxVol])
                    print(int(dist), vol)

                    volume.SetMasterVolumeLevel(vol, None)

                # Two Fingers Up Scrolling
                if fingers['4'] == 1 and fingers['8'] == 1 and fingers['12'] == 1 and fingers['16'] == 0 and fingers['20'] == 0:

                    sequence.append(landmarks.flatten())
                    sequence = sequence[-30:]  # Grab the last 30 frames

                    if len(sequence) == 30:
                        # sequence = np.expand_dims(np.reshape(sequence, (30, 63)), axis=0)
                        # print(sequence.shape)
                        # res = action(sequence)
                        # print(res)
                        res = action(np.expand_dims(sequence, axis=0))[0]
                        print(res)
                        print(actions[np.argmax(res)])

                        act = actions[np.argmax(res)]
                        sequence = []

                        if act == "scrollDown":
                            mouse.wheel(-1)

                        if act == "scrollUp":
                            mouse.wheel(1)

        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Camera", image)


def handDetection(image, handDetector):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converting to RGB because that's the color scheme mp uses
    image.flags.writeable = False
    results = handDetector.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Going back to BGR for opencv

    return image, results


def getIndexCoords(x, y):
    x, y = x * width, y * height

    return int(x), int(y)


def convertCoords(point):
    cx = int(np.interp(point[0], (padding, width - padding), (0, 1920)))
    cy = int(np.interp(point[1], (padding, height - padding), (0, 1080)))

    return cx, cy


def drawLandmarks(mpDrawing, image, hand_landmarks, mpHands):
    mpDrawing.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS,
                             mpDrawing.DrawingSpec(color=(0, 0, 216), thickness=2, circle_radius=5),
                             mpDrawing.DrawingSpec(color=(0, 204, 255), thickness=2, circle_radius=1))


def fingerRecognition(landmarks):
    fingers = {"4": 0, "8": 0, "12": 0, "16": 0, "20": 0}

    if len(landmarks) != 0:

        # print(landmarks)

        # Thumbs
        x4, y4 = getIndexCoords(landmarks[4][0], landmarks[4][1])
        x3, y3 = getIndexCoords(landmarks[3][0], landmarks[3][1])

        if x4 > x3:
            fingers["4"] = 1

        # Index finger
        x8, y8 = getIndexCoords(landmarks[8][0], landmarks[8][1])
        x6, y6 = getIndexCoords(landmarks[6][0], landmarks[6][1])

        if y8 < y6:
            fingers["8"] = 1

        # Middle finger
        x12, y12 = getIndexCoords(landmarks[12][0], landmarks[12][1])
        x10, y10 = getIndexCoords(landmarks[10][0], landmarks[10][1])

        if y12 <= y10:
            fingers["12"] = 1

        # Ring finger
        x16, y16 = getIndexCoords(landmarks[16][0], landmarks[16][1])
        x14, y14 = getIndexCoords(landmarks[14][0], landmarks[14][1])

        if y16 <= y14:
            fingers["16"] = 1

        # little finger
        x20, y20 = getIndexCoords(landmarks[20][0], landmarks[20][1])
        x18, y18 = getIndexCoords(landmarks[18][0], landmarks[18][1])

        if y20 <= y18:
            fingers["20"] = 1

    return fingers


def initVol():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    volRange = volume.GetVolumeRange()

    return volume, volRange[0], volRange[1]


def drawVolume(image, ind_x, ind_y, thumb_x, thumb_y):
    cx, cy = (ind_x + thumb_x) // 2, (ind_y + thumb_y) // 2

    cv2.circle(image, (ind_x, ind_y), 5, (255, 0, 255), cv2.FILLED)
    cv2.circle(image, (thumb_x, thumb_y), 5, (255, 0, 255), cv2.FILLED)
    cv2.line(image, (ind_x, ind_y), (thumb_x, thumb_y), (0, 255, 0), 3)
    cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)


if __name__ == "__main__":
    main()
