import sys
sys.path.append('.venv/Lib/site-packages/')

import cv2
import numpy as np

# Generate detector
myDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(myDict, detectorParams=cv2.aruco.DetectorParameters())

# Load camera calibration
calibration = np.load("notebooks/data/calibration_OBS.npz")
MARKER_LENGTH = 0.015
cameraMatrix = calibration['cameraMatrix']
distCoeffs = calibration['distCoeffs']

# Define video capture
cap = cv2.VideoCapture(1) # 0 for default camera and 1 for OBS camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, frame = cap.read()

while ok:
    print('[INFO] Capturing...')
    ok, frame = cap.read()
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
    output = frame.copy()
    output = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds, borderColor=(0, 255, 0))
    
    if markerIds is not None:
        for i in range(len(markerIds)):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners[i], MARKER_LENGTH, cameraMatrix, distCoeffs)
            output = cv2.drawFrameAxes(output, cameraMatrix, distCoeffs, rvecs, tvecs, MARKER_LENGTH)

    cv2.imshow('frame', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

if not ok:
    print('[INFO] Error reading camera')
