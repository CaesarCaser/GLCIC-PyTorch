# coding:utf-8
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np


def calc_ssim(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.

    References
    -------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    '''
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score


def calc_psnr(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    psnr_score : numpy.float64
        峰值信噪比(Peak Signal to Noise Ratio, PSNR).

    References
    -------
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    '''
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score


if __name__ == '__main__':
    input1Path = r'D:\4.Codes\GLCIC-PyTorch\datasets\evaluate\origin4.png'
    input2Path = r'D:\4.Codes\GLCIC-PyTorch\datasets\evaluate\GLCIC4.png'
    input3Path = r'D:\4.Codes\GLCIC-PyTorch\datasets\evaluate\impro4.png'
    SSIM1 = calc_ssim(input1Path, input2Path)
    SSIM2 = calc_ssim(input1Path, input3Path)
    PSNR1 = calc_psnr(input1Path, input2Path)
    PSNR2 = calc_psnr(input1Path, input3Path)
    print(SSIM1, SSIM2)
    print(PSNR1, PSNR2)
    # i = 0
    # num = 1
    # imgList = [[]]
    # for filename in os.listdir(inputPath):
    #     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    #         image_path = os.path.join(inputPath, filename)
    #         imageUnit = Image.open(image_path)
    #         imgList[i].append(imageUnit)
    #         if num % 3 == 0:
    #             print(imgList[i])
    #             i = i + 1
    #         num = num + 1
    # print(imgList)
