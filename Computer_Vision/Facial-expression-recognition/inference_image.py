import sys

import dlib
import cv2
import torch
import os
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from model_choose import CVFacialExpressionModel


#模型测试

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 人脸检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 68个特征点检测器
testsize = 48 # 输入图片尺寸

model = CVFacialExpressionModel(2) # 模型加载
model.eval() # 设置为评估模式
torch.no_grad() # 关闭梯度计算

modelpath = sys.argv[1]
model.load_state_dict(torch.load(modelpath,map_location=lambda storage, loc: storage))  # 加载模型 其中三个参数的含义 分别是：模型路径，映射到CPU还是GPU，默认是CPU

data_transforms = transforms.Compose([
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化
        ])

def preimg(im,landmarks):
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
    roi = im[ymin-int(diff_height//2):ymax+int(diff_height//2),xmin-int(diff_width//2):xmax+int(diff_width//2),0:3]
    return roi,xmin-int(diff_width//2),ymin-int(diff_height//2),dstlen

image_paths = os.listdir(sys.argv[2]) # 读取图片路径
for image_path in image_paths:
    im = cv2.imread(os.path.join(sys.argv[2],image_path),1)
    rects = cascade.detectMultiScale(im,1.3,5) # 人脸检测
    x,y,w,h = rects[0]
    rect = dlib.rectangle(x,y,x+w,y+h)
    landmarks = np.matrix([[p.x,p.y] for p in predictor(im,rect).parts()]) # 68个特征点检测

    # 预测
    roi,newx,newy,dstlen = preimg(im,landmarks) # 截取局部特征

    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB) # 转换为RGB
    roiresize = cv2.resize(roi,(testsize,testsize)) # 缩放为48x48
    imgblob = data_transforms(roiresize).unsqueeze(0) # 转换为张量
    imgblob.requires_grad = False # 关闭梯度计算
    preds = F.softmax(model(imgblob)) # 模型预测
    index = np.argmax(preds.detach().numpy()) # 预测结果
    print(preds,'label:' ,index) # 打印预测结果

    # 绘制预测框
    im_h,im_w,im_c = im.shape
    posx = int(newx + dstlen)
    posy = int(newy + dstlen)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(im,(int(newx),int(newy)),(posx,posy),(0,0,255),2)

    if index == 0:
        cv2.putText(im,'du',(int(posx),int(posy)),font,1.2,(0,0,255),2)
    elif index == 1:
        cv2.putText(im,'smile',(int(posx),int(posy)),font,1.2,(0,0,255),2)
    # elif index == 2:
    #     cv2.putText(im,'smile',(int(posx),int(posy)),font,1.2,(0,0,255),2)
    # elif index == 3:
    #     cv2.putText(im,'open',(int(posx),int(posy)),font,1.2,(0,0,255),2)

    cv2.namedWindow('result',0)
    cv2.imshow('result',im)
    cv2.imwrite(os.path.join('results',image_path),im)
    k = cv2.waitKey(0)
    if k == ord('q'):
        quit(0)

#
im = cv2.imread(sys.argv[2])
predict = model(im)



