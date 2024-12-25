import sys

import dlib
import cv2
import torch
from torchvision import transforms
from model_choose import CVFacialExpressionModel

#模型测试

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 人脸检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 68个特征点检测器
model = CVFacialExpressionModel(2) # 模型加载
model.eval() # 设置为评估模式
torch.no_grad() # 关闭梯度计算

modelpath = sys.argv[1]
model.load_state_dict(torch.load(modelpath,map_location=lambda storage, loc: storage))  # 加载模型 其中三个参数的含义 分别是：模型路径，映射到CPU还是GPU，默认是CPU


im = cv2.imread(sys.argv[2])
predict = model(im)



