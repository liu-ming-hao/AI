'''
预测模型
'''
from net import ResNet18
import torch
from torchvision import transforms
import os
import cv2
import numpy as np
from torch.nn import functional as F
from PIL import Image

# 定义全局变量
test_image_size = 224
num_classes = 20

# 模型初始化
model = ResNet18(num_classes=num_classes)
MODEL_PATH = 'models/model.pth'
model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)) 
model.eval()
torch.no_grad()

# 类别标签
CAT_LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 批量预测
def predict(image_path_dir):
    image_list = os.listdir(image_path_dir)
    for image_name in image_list:
        image_path = os.path.join(image_path_dir, image_name)
        # 使用PIL读取图像
        image = Image.open(image_path).convert('RGB')
        # 保存原始图像用于后续显示
        cv_image = cv2.imread(image_path)
        
        # 预处理图像
        input_tensor = transform(image).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            model.cuda()
            
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            predict_label = torch.sigmoid(output).cpu().numpy().squeeze()
            
        # 获取排序后的索引
        sorted_index = np.argsort(predict_label)[::-1]  # 降序排序
        
        # 在图像上绘制结果
        im_h, im_w = cv_image.shape[:2]
        pos_x = int(im_w/5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 显示前三个预测结果
        for i, idx in enumerate(sorted_index[:3]):
            text = f"{CAT_LABELS[idx]}: {predict_label[idx]:.2f}"
            y_pos = int((i + 1) * im_h/4)
            cv2.putText(cv_image, text, (pos_x, y_pos), font, 0.8, (0, 0, 255), 2)
        
        # 保存和显示结果
        if not os.path.exists('result'):
            os.makedirs('result')
        cv2.imwrite(os.path.join('result', image_name), cv_image)
        cv2.imshow('result', cv_image)
        
        k = cv2.waitKey(0)
        if k == ord('q'):
            break


if __name__ == '__main__':
    image_path_dir = 'test_images'
    predict(image_path_dir)
