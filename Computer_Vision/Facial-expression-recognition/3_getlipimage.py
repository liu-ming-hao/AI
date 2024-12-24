import os

import cv2
import dlib
import numpy as np
# 提取嘴部局部特征图片

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 人脸检测器

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 68个特征点检测器

# 关键点检测
def get_landmarks(im):
    print('start get_landmarks')
    rects = cascade.detectMultiScale(im, 1.3, 5) # 检测人脸,各参数含义：scaleFactor：检测窗口缩小的比例，越小越慢；minNeighbors：最少需要检测到多少个窗口内的目标；minSize：目标的最小尺寸；maxSize：目标的最大尺寸
    x,y,w,h = rects[0] # 取第一个人脸的坐标,各坐标含义：x,y为左上角坐标，w,h为宽和高
    rect = dlib.rectangle(x, y, x+w, y+h) # 转换成dlib格式
    print('end get_landmarks')
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()]) # 68个特征点坐标

# 提取嘴部局部特征图片
def get_lip_image(im, landmarks):
    print('start get_lip_image')
    xmin, ymin, xmax, ymax = 10000, 10000, 0, 0
    for i in range(48,67):
        x = landmarks[i,0]
        y = landmarks[i,1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y # 找出嘴部的最小外接矩形

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax,xmin:xmax,0:3]

    # 计算扩大的·roi·的大小
    if roiwidth > roiheight:
        dstlen = roiwidth * 1.5
    else:
        dstlen = roiheight * 1.5

    # 计算要增加的宽度和高度
    diff_width = dstlen - roiwidth
    diff_height = dstlen - roiheight

    print('end get_lip_image')
    # 返回截取局部特征后的图片
    return im[ymin-int(diff_height//2):ymax+int(diff_height//2),xmin-int(diff_width//2):xmax+int(diff_width//2),0:3]

# 在原图上可视化关键点
def visualize_landmarks(im, landmarks):
    print('start visualize_landmarks')
    im = im.copy() # 复制一份原图
    for idx,point in enumerate(landmarks): # 遍历每个特征点
        pos = (point[0,0], point[0,1]) # 转换成cv2格式
        cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255)) # 标注特征点编号
        cv2.circle(im, pos, 3, color=(0,255,255)) # 标注特征点

    print('end visualize_landmarks')
    return im

# 遍历数据集主函数
def process_dataset(dataset_path):
    list_dirs = os.walk(dataset_path) # 遍历数据集目录
    for root, dirs, files in list_dirs:
        print('root:' + root)
        for d in dirs:
            print(os.path.join(root,d))
        for f in files:
            fileid = f.split('.')[0]
            filetype = f.split('.')[1]
            filepath = os.path.join(root,f)
            print('f:' + filepath)
            im = cv2.imread(filepath, 1) # 读取图片,参数1表示读取3通道彩色图片
            os.remove(filepath)
            landmarks = get_landmarks(im) # 提取特征点
            show_landmarks = visualize_landmarks(im, landmarks) # 可视化特征点
            roi_image = get_lip_image(im, landmarks) # 提取嘴部局部特征图片
            roi_path = filepath.replace('.' + filetype,'_mouth.png')
            if '0043' in fileid:
                cv2.imshow('keypoints', show_landmarks) # 显示局部特征图片
                cv2.waitKey(0) # 等待按键

            cv2.imwrite(roi_path, roi_image)


if __name__ == '__main__':
    dataset_path = 'source_data'
    process_dataset(dataset_path)
    print('Done!')