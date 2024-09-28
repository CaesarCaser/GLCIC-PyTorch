import cv2
import numpy as np
import os
from torchvision import transforms
import copy
from PIL import Image

reinforceTimes = 2


# 镜像
def imageFlip(inputImage):
    flipped_image = np.fliplr(inputImage)
    return flipped_image


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 高斯噪声
def addGaussianNoise(image, percentage):
    G_Noising = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percentage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noising[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noising


# 共402张图片，目标增强1206张，分别以旋转、镜像、高斯噪声进行处理
def dataReinForce(inPutPath, outPutPath):
    i = "1"
    for filename in os.listdir(inPutPath):
        image_path = os.path.join(inPutPath, filename)
        img = cv2.imread(image_path)
        cv2.imwrite(outPutPath + i + '.png', img)
        i = str(int(i) + 1)
        flippedImage = imageFlip(img)
        cv2.imwrite(outPutPath + i + '.png', flippedImage)
        i = str(int(i) + 1)
        rotateImage = rotate(img, 180)
        cv2.imwrite(outPutPath + i + '.png', rotateImage)
        i = str(int(i) + 1)
        gaussImage = addGaussianNoise(img, 0.1)
        cv2.imwrite(outPutPath + i + '.png', gaussImage)
        i = str(int(i) + 1)
        print(i)


if __name__ == '__main__':
    inputPath = r'D:\1.Data\screenIMG'
    outputPath = r'D:\1.Data\screenReinforIMG'
    dataReinForce(inputPath, outputPath)
