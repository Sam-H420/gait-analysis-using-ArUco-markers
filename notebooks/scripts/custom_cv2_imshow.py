import cv2
import matplotlib.pyplot as plt

def cv2_imshow(cv2image):
    """Takes an cv2 image and displays it"""
    plt.imshow(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis('off')
    plt.show()