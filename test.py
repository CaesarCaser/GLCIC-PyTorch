import cv2
import numpy as np
from PIL import Image


# image = cv2.imread('2.jpg')
# channels = image.shape[2]
# if channels == 3:
#     print("RGB")
# elif channels == 4:
#     print("RGBA")

mpv = np.zeros(shape=(3,))
print(mpv.shape)
img = Image.open('2.jpg')
img1 = Image.open('000001.jpg')
x = np.array(img) / 255.
x1 = np.array(img1) / 255.
mpv += x1.mean(axis=(0, 1))
print(mpv)


