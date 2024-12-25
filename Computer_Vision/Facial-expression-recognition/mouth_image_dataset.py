import torch
from PIL import Image # PIL库,它的作用是 打开图像文件，并返回一个Image对象
from torchvision import transforms
from torch.utils.data import *
# torchvision是pytorch的一个子模块，它提供了很多关于计算机视觉的工具，包括数据集、模型、预训练模型等,transforms是torchvision中的一个子模块，它提供了很多关于图像变换的工具，包括裁剪、缩放、旋转、翻转、归一化等

# 数据集读取

class MouthImageDataset(Dataset):
    def __init__(self, datalist, data_transformer):
        datalines = open(datalist,'r').readlines()
        self.samples = []
        self.data_transformer = data_transformer
        for data in datalines:
            data_path,data_label = data.strip().split(' ')
            self.samples.append([data_path,data_label])

    def __getitem__(self, index):
        data_path, data_label = self.samples[index]
        image = Image.open(data_path)
        image = self.data_transformer(image) # 对图像进行预处理
        return image, int(data_label)

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    # 数据预处理 - 数据增强
    train_transformer = transforms.Compose([
        transforms.RandomResizedCrop(60), # 随机裁剪
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomVerticalFlip(), # 随机垂直翻转
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化
    ])

    val_transformer = transforms.Compose([
        transforms.Resize(60), # 调整大小
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化
    ])

    train_dataset = MouthImageDataset('1_train.txt', train_transformer)
    print('train_dataset length:', len(train_dataset))
    print('train_dataset size:', str(train_dataset.__len__()))

    image,label = train_dataset.__getitem__(0)
    print('traindataset first image size:', str(image.shape))
    print('traindataset first image label:', str(label))

    train_data_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=True)

    count = 0
    for image,label in train_data_loader:
        print('第' + str(count) + '个batch')
        count += 1
        print('batch image size:', str(image.shape))
        print('batch image label:', str(label))
    #val_dataset = MouthImageDataset('val.txt', val_transformer)