import os

import cv2
from PIL.Image import Image

from dataTransfrom import batchImageTransform
from imageSharp import guideFilter
from dataReinforcement import dataReinForce

# 先图像裁剪，然后图像导向滤波，再然后图像增
if __name__ == '__main__':
    # inputPath = r'E:\GLCIC\GLCIC-PyTorch\datasets\Tang\origin-Tang'
    # outputPath = r'E:\GLCIC\GLCIC-PyTorch\datasets\Tang\TransformFlow\transformed'
    # 裁剪
    # batchImageTransform(inputPath, outputPath)
    # 滤波
    # directory_to_search = r'E:\GLCIC\GLCIC-PyTorch\datasets\Tang\outPut'
    # filterOutputPath = r'E:\GLCIC\GLCIC-PyTorch\datasets\Tang\TransformFlow\guideFiltered'
    # eps = 0.01
    # winSize = (16, 16)  # 类似卷积核（数字越大，磨皮效果越好）
    # i = "0"
    #
    # for filename in os.listdir(r'E:\GLCIC\GLCIC-PyTorch\datasets\Tang\outPut'):
    #     # 匹配图片文件扩展名
    #     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    #         i = str(int(i) + 1)
    #         image_path = os.path.join(directory_to_search, filename)
    #         # 打开图片
    #         image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    #         image = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
    #         I = image / 255.0  # 将图像归一化
    #         p = I
    #         s = 3  # 步长
    #         guideFilter(I, p, winSize, eps, s, filterOutputPath, i)
    # 增强
    reinInputPath = r'E:\GLCIC\GLCIC-PyTorch\datasets\Tang\TransformFlow\guideFiltered'
    reinOutputPath = r'E:\GLCIC\GLCIC-PyTorch\datasets\Tang\TransformFlow\reinforcement'
    dataReinForce(reinInputPath, reinOutputPath)


