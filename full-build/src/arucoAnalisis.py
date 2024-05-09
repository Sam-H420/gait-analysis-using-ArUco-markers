import cv2
import numpy as np
import matplotlib.pyplot as plt

class Pose:
    """Define a pose object with rotation and translation vectors relative to an origin point"""
    def __init__(self, rvec: list[list[list[float]]], tvec: list[list[list[float]]]):
        self.__rvec = rvec
        self.__tvec = tvec

    @property
    def rvec(self):
        return self.__rvec
    
    @property
    def tvec(self):
        return self.__tvec

    def __str__(self):
        return f'Pose(rvec={self.__rvec}, tvec={self.__tvec})'

    def __repr__(self):
        return str(self)
    
class CameraCalibration:
    def __init__(self, camera_matrix: list[list[float]], dist_coeffs: list[list[float]]):
        self.__camera_matrix = np.array(camera_matrix)
        self.__dist_coeffs = np.array(dist_coeffs)

    @property
    def camera_matrix(self):
        return self.__camera_matrix
    
    @property
    def dist_coeffs(self):
        return self.__dist_coeffs

    def __str__(self):
        return f'Camera matrix: {self.__camera_matrix}\nDistortion coefficients: {self.__dist_coeffs}'
    
    def __repr__(self):
        return str(self)

class Camera:
    """Camera Class"""

    def __init__(self, id: int, reference: str, calibration: CameraCalibration | None = any):
        self.__id = id
        self.__reference = reference
        self.__calibration = calibration

    @property
    def id(self):
        return self.__id
    
    @property
    def reference(self):
        return self.__reference
    
    @property
    def calibration(self):
        return self.__calibration
    
    def calibrate_camera(self, imgs: list, board: cv2.aruco.CharucoBoard):
        gray = any
        
        boardDetector = cv2.aruco.CharucoDetector(board)
        allCorners = []
        allIds = []
        allImgPoints = []
        allObjPoints = []

        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners, ids, _, _ = boardDetector.detectBoard(gray)
            objPoints, imgPoints = board.matchImagePoints(corners, ids)
            
            if ids is not None:
                allCorners.append(corners[0])
                allIds.append(ids)
                allImgPoints.append(imgPoints)
                allObjPoints.append(objPoints)

            else:
                print("Board not found in image")
                break

        allCorners = np.array(allCorners)
        allIds = np.array(allIds)
        allImgPoints = np.array(allImgPoints)
        allObjPoints = np.array(allObjPoints)

        cameraMatrix, distCoeffs = np.array([]), np.array([])
        rvecs, tvecs = np.array([]), np.array([])

        _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(allObjPoints, allImgPoints, gray.shape[::-1], cameraMatrix, distCoeffs, rvecs, tvecs)
        __calibration = CameraCalibration(cameraMatrix, distCoeffs)
        return CameraCalibration(cameraMatrix, distCoeffs), Pose(rvecs, tvecs)
    
    def __str__(self):
        return f'ID: {self.__id}\nReference: {self.__reference}\nCalibration: {self.__calibration}'


class Marker:
    """Marker Class"""

    def __init__(self, dictionary: int, id: int, corners: list[list[list[float]]], poses: list[Pose] | None = [], size = 500):
        self.__dictionary = dictionary
        self.__id = id
        self.__corners = corners
        self.__poses = poses
        self.__size = size

    @property
    def dictionary(self):
        return self.__dictionary
    
    @property
    def id(self):
        return self.__id
    
    @property
    def corners(self):
        return self.__corners
    
    @property
    def poses(self):
        return self.__poses
    
    @property
    def size(self):
        return self.__size

    @property
    def img(self):
        """Generate the marker image"""
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(self.dictionary), int(self.__id), int(self.__size))
        return marker
    
    # @poses.setter
    # def poses(self, poses: list[Pose] | None, pose: Pose | None):
    #     if poses is not None:
    #         self.__poses = poses

    #     elif pose is not None:
    #         self.__poses.append(pose)

    def __str__(self):
        return f'Marker object with dictionary: {self.__dictionary}'

class Landmark(Marker):
    """Landmark class"""
    def __init__(self, id: int, marker: Marker, relativeLocation: list[list[list[float]]] | None = None, absoluteLocation: list[list[list[float]]] | None = any, absolutePoses: list[Pose] | None = [], relativePose: Pose | None = any):
        super().__init__(marker.dictionary, marker.id, marker.corners, marker.poses, marker.size)
        self.__id = id
        self.__marker = marker
        self.__relativeLocation = relativeLocation
        self.__AbsoluteLocation = absoluteLocation
        self.__AbsolutePoses = absolutePoses
        self.__relativePose = relativePose

    @property
    def id(self):
        return self.__id

    @property
    def marker(self):
        return self.__marker
    
    @property
    def relativePose(self):
        return self.__relativePose
    
    @property
    def relativeLocation(self):
        return self.__relativeLocation
    
    @property
    def absolutePoses(self):
        return self.__AbsolutePoses
    
    @absolutePoses.setter
    def absolutePoses(self, poses: list[Pose] | None):
        if poses is not None:
            self.__AbsolutePoses = poses

        else:
          self.__AbsolutePoses = []
    
    def _rotate_vector(self, vector, rotation_vector):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rotation_matrix = np.transpose(rotation_matrix)
        rotated_vector = np.dot(vector, rotation_matrix)
        return rotated_vector
    
    def _undo_rotation(self, vector, rotation_vector):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rotation_matrix = np.transpose(rotation_matrix)
        derotated_vector = np.dot(vector, rotation_matrix)
        return derotated_vector
    
    def determine_absolute_poses(self):
        """Determine the absolute pose of the landmark"""
        if self.__relativeLocation is not None:
            for pose in self.marker.poses:
                rel_tvec = self._rotate_vector(self.__relativeLocation, pose.rvec)
                rel_rvec = [[[0.0, 0.0, 0.0]]]

                self.__relativePose = Pose(rel_rvec, rel_tvec)

                abs_tvec = pose.tvec + rel_tvec
                abs_rvec = pose.rvec + self.__relativePose.rvec

                self.__AbsolutePoses.append(Pose(abs_rvec, abs_tvec))
        else:
            print("Cannot determine the absolute poses of the landmark without a relative location.")

    def determine_relative_location(self):
        """Determine the relative pose of the landmark"""
        if self.__relativeLocation is None:
            self.__relativeLocation = self._undo_rotation((self.__AbsoluteLocation - self.marker.poses[-1].tvec), self.marker.poses[-1].rvec)
        else:
            print("The relative location has already been given.")

    def __str__(self):
        return f'Landmark poses: {self.__AbsolutePoses}'

class Segment:
    """Defines a segment composed by a Marker and a series of Landmarks"""
    def __init__(self, id: int, name: str, marker: Marker, landmarks: list[Landmark] | None = []):
        self.__id = id
        self.__name = name
        self.__marker = marker
        self.__landmarks = landmarks

    @property
    def id(self):
        return self.__id
    
    @property
    def name(self):
        return self.__name

    @property
    def marker(self):
        return self.__marker
    
    @property
    def landmarks(self):
        return self.__landmarks
    
    def add_landmark(self, relativeLocation: list[list[list[float]]]):
        """Add a landmark to the segment"""
        landmark = Landmark(len(self.__landmarks), self.__marker, relativeLocation)
        self.__landmarks.append(landmark)

    def add_landmark_from_absoluteLocation(self, pose: Pose):
        """Add a landmark to the segment from an absolute location"""
        landmark = Landmark(len(self.__landmarks), self.__marker, absoluteLocation=pose.tvec)
        landmark.determine_relative_location()
        self.__landmarks.append(landmark)
   
    def draw_landmarks(self, image: cv2.typing.MatLike, calibration: CameraCalibration):
        """Draw the landmarks on the image"""
        out = image.copy()
        for landmark in self.__landmarks:
            landmark.determine_absolute_poses()
            out = cv2.drawFrameAxes(out, calibration.camera_matrix, calibration.dist_coeffs, landmark.absolutePoses[-1].rvec, landmark.absolutePoses[-1].tvec, 0.001)
        return out
    
    def clear_landmarks(self):
        """Clear the landmarks from the segment"""
        self.__landmarks = []

    def __str__(self):
        return f'Segment object with ID: {self.__id}'
    
class Toolset:
    """Define a set of tools for image processing"""

    def __init__(self, dictionary: int):
        self.__dictionary = dictionary

    @property
    def dictionary(self):
        return self.__dictionary

    def generate_marker_img(self, marker_id, marker_size = 500):
        """Generate a marker with the given ID"""
        img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(self.dictionary), marker_id, marker_size)
        return img
    
    def generate_board(self, squares: tuple[int, int] = (5,5), square_length: float = 0.04, marker_length: float = 0.02, page_length: int = 640, margin: int = 20):
        """Generate a board with the given parameters"""
        board = cv2.aruco.CharucoBoard((squares[0], squares[1]), square_length, marker_length, cv2.aruco.getPredefinedDictionary(self.dictionary))
        size_ratio = squares[0] / squares[1]
        img = cv2.aruco.CharucoBoard.generateImage(board, (page_length, int(page_length*size_ratio)), marginSize=margin)
        return board, img

    def detect_markers_calibration(self, img: cv2.typing.MatLike, cameraCalibration: CameraCalibration | None = None, marker_length: float = 0.02):
        """Detect markers in the given image"""
        ids = np.array([])
        corners = np.array([])

        marker_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(marker_dict, parameters)
        corners, ids, _ = detector.detectMarkers(img)

        output = img.copy()
        output = cv2.aruco.drawDetectedMarkers(output, corners, ids, borderColor=(0, 255, 0))

        markers: list = []

        for i in range(np.size(ids)):
            
            if cameraCalibration is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_length, cameraCalibration.camera_matrix, cameraCalibration.dist_coeffs)
                output = cv2.drawFrameAxes(output, cameraCalibration.camera_matrix, cameraCalibration.dist_coeffs, rvecs, tvecs, marker_length)
                pose = Pose(rvecs, tvecs)
                marker = Marker(self.dictionary, ids[i], corners[i], [pose])

            markers.append(marker)

        return markers, output
    
    def detect_markers_run(self, img: cv2.typing.MatLike, markers: list[Marker], cameraCalibration: CameraCalibration | None = None, marker_length: float = 0.02):
        """Detect markers in the given image"""
        ids = np.array([])
        corners = np.array([])

        marker_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(marker_dict, parameters)
        corners, ids, _ = detector.detectMarkers(img)

        output = img.copy()
        output = cv2.aruco.drawDetectedMarkers(output, corners, ids, borderColor=(0, 255, 0))

        for i in range(np.size(ids)):
            if ids[i] in [marker.id for marker in markers]:
                if cameraCalibration is not None:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_length, cameraCalibration.camera_matrix, cameraCalibration.dist_coeffs)
                    output = cv2.drawFrameAxes(output, cameraCalibration.camera_matrix, cameraCalibration.dist_coeffs, rvecs, tvecs, marker_length)
                    pose = Pose(rvecs, tvecs)
                    for marker in markers:
                        if marker.id == ids[i]:
                            marker.poses.append(pose)

        return markers, output

    def __str__(self):
        return f'Toolset object with dictionary: {self.__dictionary}'

def cv2_imshow(cv2image):
    """Takes an cv2 image and displays it (useful for Jupyter Notebooks)"""
    plt.imshow(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis('off')
    plt.show()