import sys
sys.path.append('.venv/Lib/site-packages/')

import cv2
import numpy as np

# Create a Charuco board
# taken from https://medium.com/@ed.twomey1/using-charuco-boards-in-opencv-237d8bc9e40d

# ENTER YOUR PARAMETERS HERE:
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 640   # total length of the page in pixels
MARGIN_PX = 20    # size of the margin in pixels
SAVE_NAME = 'calibration_test'

def create_new_board():
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, DICTIONARY)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    cv2.imwrite(f'notebooks\img\{SAVE_NAME}.png', img)
    return board

charuco_board = create_new_board()

# Load calibration video
cap = cv2.VideoCapture("notebooks/vid/test_calibration.mp4") # 0 for default camera and 1 for OBS camera, for file use cv2.VideoCapture('file.mp4')

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

board_detector = cv2.aruco.CharucoDetector(charuco_board)
allCorners = []
allIds = []
allImgPoints = []
allObjPoints = []

for i in range(len(imgs)):
    print(f'[INFO] Detecting board in frame {i}')
    gray = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
    corners, ids, _, _ = board_detector.detectBoard(gray)
    objPoints, imgPoints =  charuco_board.matchImagePoints(corners, ids)
    
    if ids is not None:
        allCorners.append(corners[0])
        allIds.append(ids)
        allImgPoints.append(imgPoints)
        allObjPoints.append(objPoints)

print('[INFO] Calibrating camera')
        
allCorners = np.array(allCorners)
allIds = np.array(allIds)
allImgPoints = np.array(allImgPoints)
allObjPoints = np.array(allObjPoints)

cameraMatrix, distCoeffs = np.array([]), np.array([])
rvecs, tvecs = np.array([]), np.array([])

try:
    _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(allObjPoints, allImgPoints, gray.shape[::-1], cameraMatrix, distCoeffs, rvecs, tvecs)
    print(cameraMatrix)
    np.savez(f'notebooks\data\{SAVE_NAME}.npz', cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

except:
    print('[ERROR] Error calibrating camera')
    sys.exit()