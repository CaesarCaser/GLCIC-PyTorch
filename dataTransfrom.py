import os
from torchvision import transforms
from PIL import Image
import cv2


# 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
transform = transforms.Compose([
    transforms.Resize((218, 178)),
    transforms.ToTensor()
])




def singleImageTransform(imgPath, imgName):
    image_path = os.path.join(imgPath, imgName)
    img = Image.open(image_path)
    # 计算通道数，筛选RGBA图片，将其转换为RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transformedImg = transform(img)
    transformedImg = transforms.ToPILImage()(transformedImg)
    output_file_path = os.path.join(imgPath, 'transformed' + imgName[-5] + '.png')
    transformedImg.save(output_file_path)


def batchImageTransform(inputPath, outputPath):
    i = "0"
    # r'E:\GLCIC\GLCIC-PyTorch\datasets\Tang\origin-Tang'
    for filename in os.listdir(inputPath):
        # 匹配图片文件扩展名
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            i = str(int(i) + 1)
            image_path = os.path.join(directory_to_search, filename)
            # 打开图片
            img_as_img = Image.open(image_path)
            # 计算通道数，筛选RGBA图片，将其转换为RGB
            if img_as_img.mode != 'RGB':
                img_as_img = img_as_img.convert('RGB')
            # 应用转换
            transformed_img = transform(img_as_img)
            # 将Tensor转换回PIL Image
            transformed_img = transforms.ToPILImage()(transformed_img)
            # 这里可以对img_as_img进行进一步处理，例如保存到列表、数组或直接用于模型
            output_file_path = os.path.join(outputPath, i + '_1.png')
            transformed_img.save(output_file_path)
            print(i)


if __name__ == '__main__':
    # path = r'E:\GLCIC\GLCIC-PyTorch\images\transform'
    # imageName = 'guideFilter4.png'
    # 定义文件夹路径
    directory_to_search = r'D:\1.Data\temp'  # 替换为你的文件夹路径
    output_directory = r'D:\1.Data\temp'  # 替换为你的输出文件夹路径
    batchImageTransform(directory_to_search, output_directory)
