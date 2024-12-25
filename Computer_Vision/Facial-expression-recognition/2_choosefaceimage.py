import os
import sys

import cv2

# 数据预处理二： 保留有人脸的数据

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 加载级联分类器

def detect_face(file_path):
    images = os.listdir(file_path)

    for image in images:
        im = cv2.imread(os.path.join(file_path,image),1) # 读取图像
        rects = cascade.detectMultiScale(im,1.3,5) # 检测人脸，参数说明：
        # im：待检测的图像
        # 1.3：图像缩放比例，1.3表示每次缩小30%
        # 5：最小人脸大小
        print('detected %d faces'%len(rects))   # 打印检测到的人脸数
        if len(rects) == 0:
            os.remove(os.path.join(file_path,image))

if __name__ == '__main__':
    detect_face('source_data/smile')