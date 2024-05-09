import sys
sys.path.append('.venv/Lib/site-packages/')

import cv2
import numpy as np

# Generate detector
myDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(myDict, detectorParams=cv2.aruco.DetectorParameters())

# Load camera calibration
calibration = np.load("notebooks/data/calibration_test.npz")
MARKER_LENGTH = 0.015
cameraMatrix = calibration['cameraMatrix']
distCoeffs = calibration['distCoeffs']

# Load video
cap = cv2.VideoCapture("notebooks/vid/test_01.mp4") # 0 for default camera and 1 for OBS camera, for file use cv2.VideoCapture('file.mp4')

if not cap.isOpened():
    print('[ERROR] Error opening video')
    sys.exit()

count = 0
imgs = []
gray = any

while (cap.isOpened()):
    print(f'[INFO] Reading frame {count}')
    ret, frame = cap.read()

    if ret:
        imgs.append(frame)
        count += 1
        continue
    else:
        break

cap.release()

try:
    cv2.imshow('frame', imgs[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except:
    print('[ERROR] Error showing image')
    sys.exit()

height, width, layers = imgs[0].shape
size = (width,height)

out = cv2.VideoWriter('notebooks/vid/detected_test_01.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, size) # 30 fps
 
for i in range(len(imgs) - 1):
    print(f'[INFO] Detecting markers in frame {i}')
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(imgs[i])
    output = imgs[i].copy()
    output = cv2.aruco.drawDetectedMarkers(output, markerCorners, markerIds, borderColor=(0, 255, 0))
    
    if markerIds is not None:
        for j in range(len(markerIds)):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners[j], MARKER_LENGTH, cameraMatrix, distCoeffs)
            output = cv2.drawFrameAxes(output, cameraMatrix, distCoeffs, rvecs, tvecs, MARKER_LENGTH)

    out.write(output)

out.release()