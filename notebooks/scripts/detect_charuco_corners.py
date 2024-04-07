import sys
sys.path.append('C:/Sam/UTB/2024-1/Vision/gait-analysis-using-aruco-markers/.venv/Lib/site-packages/')

import cv2
import numpy as np

SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 640   # total length of the page in pixels
MARGIN_PX = 20    # size of the margin in pixels
SAVE_NAME = 'charuco_board.png'
# ------------------------------

def create_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    cv2.imwrite(f'img/{SAVE_NAME}', img)
    return board
    

charuco_board = create_new_board()

myDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.CharucoDetector(charuco_board)
gray = any

cap = cv2.VideoCapture(1) # 0 for default camera and 1 for OBS camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, frame = cap.read()

while ok:
    corners = np.array([])
    ids = np.array([])
    markerIds = np.array([])
    markerCorners = np.array([])
    
    print('[INFO] Capturing...')
    ok, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, markerCorners, markerIds = detector.detectBoard(gray, corners, ids, markerCorners, markerIds)
    output = frame.copy()
    output = cv2.aruco.drawDetectedCornersCharuco(output, corners, ids)
    
    cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

if not ok:
    print('[INFO] Error reading camera')