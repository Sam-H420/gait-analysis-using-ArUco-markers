import sys
sys.path.append('C:/Sam/UTB/2024-1/Vision/gait-analysis-using-aruco-markers/.venv/Lib/site-packages/')

import cv2
import numpy as np

myDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(myDict, detectorParams=cv2.aruco.DetectorParameters())

cap = cv2.VideoCapture(1) # 0 for default camera and 1 for OBS camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, frame = cap.read()

while ok:
    markerIds = np.array([])
    markerCorners = np.array([])
    rejectedCandidates = np.array([])
    
    print('[INFO] Capturing...')
    ok, frame = cap.read()
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame, markerCorners, markerIds, rejectedCandidates)
    output = frame.copy()
    output = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds, borderColor=(0, 255, 0))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.imwrite('a.png', output)
        break

if not ok:
    print('[INFO] Error reading camera')