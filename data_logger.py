import os
import cv2
import numpy as np
import mediapipe as mp

width = 640
height = 480
dataPath = os.path.join("Data")
actions = np.array(['scrollDown', 'scrollUp', 'increaseVol', 'decreaseVol'])
noSequences = 30
sequenceLength = 30


def main():
    # Initializing Camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    dirSetup()
    mpDrawing = mp.solutions.drawing_utils

    # Loading the Model
    mpHands = mp.solutions.hands
    with mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:

        for action in actions:
            for sequence in range(noSequences):
                for frameNum in range(sequenceLength):
                    print(action, sequence, frameNum)

                    ret, frame = cam.read()
                    image = cv2.flip(frame, 1)
                    image, results = handDetection(image, hands)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            drawLandmarks(mpDrawing, image, hand_landmarks, mpHands)

                            if frameNum == 0:
                                cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                            (15, 12),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                # Show to screen
                                cv2.imshow('Data Logger', image)
                                cv2.waitKey(1000)
                            else:
                                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                            (15, 12),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                # Show to screen
                                cv2.imshow("Data Logger", image)

                            keyPoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
                            npy_path = os.path.join(dataPath, action, str(sequence), str(frameNum))
                            np.save(npy_path, keyPoints)
                            print(keyPoints.shape)
                    else:
                        keyPoints = np.zeros([21, 3])
                        npy_path = os.path.join(dataPath, action, str(sequence), str(frameNum))
                        np.save(npy_path, keyPoints)
                        cv2.imshow("Data Logger", image)

                    if cv2.waitKey(1) == ord("q"):
                        break

        cam.release()
        cv2.destroyAllWindows()


def handDetection(image, handDetector):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converting to RGB because that's the color scheme mp uses
    image.flags.writeable = False
    results = handDetector.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Going back to BGR for opencv

    return image, results


def drawLandmarks(mpDrawing, image, hand_landmarks, mpHands):
    mpDrawing.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS,
                             mpDrawing.DrawingSpec(color=(0, 0, 216), thickness=2, circle_radius=5),
                             mpDrawing.DrawingSpec(color=(0, 204, 255), thickness=2, circle_radius=1))


def dirSetup():
    for action in actions:
        for sequence in range(noSequences):
            try:
                os.makedirs(os.path.join(dataPath, action, str(sequence)))
            except:
                pass


if __name__ == "__main__":
    main()
