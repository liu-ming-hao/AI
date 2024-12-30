''' 
    第二步： 构建数据集，将数据集分为训练集和验证集。
'''

import os
from PIL import Image
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from data_first_getlabel import load_img_name_list,load_cls_labels_from_npy

IMG_FOLDER_NAME = 'JPEGImages'

class VOCDataset(Dataset):
    def __init__(self,img_list,label_list,data_root,data_transform):
        self.samples = []
        self.transforms = data_transform

        for img_name,label in zip(img_list,label_list):
            image_path = os.path.join(data_root,IMG_FOLDER_NAME,img_name+'.jpg')
            self.samples.append((image_path,label))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        image_path,label = self.samples[index]
        image = Image.open(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image,label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default='D:\pythoncode\muke\\17\multi_label\multi_label\VOC2012',help='the root of the dataset')
    parser.add_argument('--dataset_path_train',type=str,default='D:\pythoncode\muke\\17\multi_label\multi_label\VOC2012\ImageSets\Main\\train.txt',help='the path of the train dataset')
    parser.add_argument('--out',type=str,default='cls_labels.npy')
    args = parser.parse_args()

    # 数据增强
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    # 获取训练集和验证集图片名称列表
    train_img_name_list = load_img_name_list(args.dataset_path_train)

    # 获取训练集标签
    cls_labels_dict = np.load(args.out,allow_pickle=True).item()
    train_label_list = load_cls_labels_from_npy(cls_labels_dict,train_img_name_list)

    # 构建训练集数据集
    train_dataset = VOCDataset(train_img_name_list,train_label_list,args.data_root,data_transform =data_transform)

    # 打印相关信息
    print(f'train_dataset length: {train_dataset.__len__()}')
    print(f'train_dataset sample: {train_dataset.__getitem__(0)}')
    image,label = train_dataset.__getitem__(0)
    print(f'image0 shape: {image.shape}')
    print(f'label0 shape: {label.shape}')
    print(f'label0: {label}')
    print(f'label0 type: {type(label)}')



