import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL.ImageFilter import FIND_EDGES, EDGE_ENHANCE, EDGE_ENHANCE_MORE, SHARPEN
# import skimage.filters as af
# import skimage.filters
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((218, 178)),
    transforms.ToTensor()
])


def laplacian_sharpen(imagePath, image, k=1):
    # 计算拉普拉斯图像
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    # 增强图像
    sharpenImg = cv2.addWeighted(image, 1, laplacian, k, 0)
    sharpenImg = transforms.ToPILImage()(sharpenImg)
    output_file_path = os.path.join(imagePath, 'sharpened' + imgName[-5] + '.png')
    sharpenImg.save(output_file_path)


def show_img(ax_img,img,title,cmap="gray"):
    ax_img.imshow(img, cmap)
    ax_img.set_title(title)
    ax_img.set_axis_off()


# 方法2：对参数radius与amount取不同的值，对比显示实验结果；
# def fun_02():
#     im_upsharp_1 = skimage.filters.unsharp_mask(img, radius=1.0, amount=100.0, multichannel=False, preserve_range=False)
#     im_upsharp_2 = skimage.filters.unsharp_mask(img, radius=2.0, amount=50.0, multichannel=False, preserve_range=False)
#     im_upsharp_3 = skimage.filters.unsharp_mask(img, radius=10.0, amount=80.0, multichannel=False, preserve_range=False)
#     fig, (ax_img, im1, im2, im3) = plt.subplots(1, 4)
#     # 显示图像
#     show_img(ax_img, img, "原始图像")
#     show_img(im1, im_upsharp_1, "im_upsharp_1")
#     show_img(im2, im_upsharp_2, "im_upsharp_2")
#     show_img(im3, im_upsharp_3, "im_upsharp_3")
#     plt.show()


def fun_03(im):
    # 导入图片
    # img_01 = cv2.imread(imagePath)
    # 转换灰度
    # im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # im = Image.open(imagePath)
    im_01 = im.filter(FIND_EDGES)
    im_02 = im.filter(EDGE_ENHANCE)
    im_03 = im.filter(EDGE_ENHANCE_MORE)
    im_04 = im.filter(SHARPEN)
    fig, (ax_img, im1, im2, im3, im4) = plt.subplots(1, 5)
    # 显示图像
    show_img(ax_img, im, "原始图像")
    show_img(im1, im_01, "总数=10")
    show_img(im2, im_02, "总数=20")
    show_img(im3, im_03, "总数=30")
    show_img(im4, im_04, "总数=40")
    plt.show()


def guideFilter(I, p, winSize, eps, s, outputPath, index):
    # 输入图像的高、宽
    h, w = I.shape[:2]

    # 缩小图像
    size = (int(round(w * s)), int(round(h * s)))
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)

    # 缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X * s)), int(round(X * s)))

    # I的均值平滑 p的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    mean_small_p = cv2.blur(small_p, small_winSize)

    # I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I * small_I, small_winSize)
    mean_small_Ip = cv2.blur(small_I * small_p, small_winSize)

    # 方差、协方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a * mean_small_I

    # 对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)

    # 放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

    q = mean_a * I + mean_b
    # q = np.clip(q, 0, 1)
    # # 乘以 255 并转换为 uint8
    # guideFilter_img = (q * 255).round().astype(np.uint8)
    # 如果 I 是彩色图像，确保 guideFilter_img 也是彩色的
    # if len(I.shape) == 3 and I.shape[2] == 3:
    #     guideFilter_img = cv2.cvtColor(guideFilter_img, cv2.COLOR_GRAY2BGR)
    guideFilter_img = q
    # 保存导向滤波结果
    guideFilter_img = guideFilter_img * 255  # (0,1)->(0,255)
    guideFilter_img[guideFilter_img > 255] = 255  # 防止像素溢出
    guideFilter_img = np.round(guideFilter_img)
    guideFilter_img = guideFilter_img.astype(np.uint8)
    # guideFilter_img = transforms.ToPILImage()(guideFilter_img)
    output_file_path = os.path.join(outputPath, 'guideFilter' + index + '.png')
    # guideFilter_img.save(output_file_path)
    cv2.imwrite(output_file_path, guideFilter_img)


if __name__ == '__main__':
    imgPath = r'D:\1.Data\temp'
    outputPath = r'D:\1.Data\temp'
    imgName = '3.jpg'
    image_path = os.path.join(imgPath, imgName)
    eps = 0.005
    winSize = (16, 16)  # 类似卷积核（数字越大，磨皮效果越好）
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    image = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
    i = imgName[-5]
    I = image / 255.0  # 将图像归一化
    p = I
    s = 3  # 步长
    guideFilter(I, p, winSize, eps, s, outputPath, i)

    # guideFilter_img = transforms.ToPILImage()(guideFilter_img)
    # 导向滤波后再进行锐化
    # fun_03(guideFilter_img)
    # cv2.imshow("image", image)
    # cv2.imshow("winSize_16", guideFilter_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # fun_03(image_path)
    # img = cv2.imread(image_path)
    # laplacian_sharpen(imgPath, img)
