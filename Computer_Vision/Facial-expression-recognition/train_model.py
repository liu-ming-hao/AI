import os
from os import write

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import *
from model_choose import CVFacialExpressionModel # 导入模型
from mouth_image_dataset import MouthImageDataset # 导入数据集
from torch.utils.tensorboard import SummaryWriter # 导入tensorboard
writer = SummaryWriter('runs') # 创建一个SummaryWriter对象，用于记录训练过程中的数据

# 使用wandb记录训练过程
import wandb
import datetime
run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")     # 记录当前时间
wandb.init(project="facial-expression-recognition",config={
    'learning_rate': 0.001,
    'epochs': 5,
    'batch_size': 16
},name=run_time)

import time

import tqdm # 进度条
#模型训练

def train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                #step_lr_scheduler.step() # 学习率调度器
                model.train(True) # 训练模式
            else:
                model.train(False) # 测试模式

            running_loss = 0.0  # 损失
            running_accs = 0.0  # 准确率-精度
            number_batch = 0  # 批次

            for data in tqdm.tqdm(dataloaders[phase]):
                #time.sleep(0.1) # 延时0.01秒，防止进度条卡顿
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad() # 梯度清零
                outputs = model(inputs) # 前向传播
                _, preds = torch.max(outputs.data, 1) # 获取预测结果
                loss = criterion(outputs, labels) # 计算损失
                if phase == 'train':
                    loss.backward() # 反向传播
                    optimizer.step() # 更新参数
                running_loss += loss.item() # 累计损失
                running_accs += torch.sum(preds == labels).item() # 累计准确率
                number_batch += 1 # 累计批次

            epoch_loss = running_loss / number_batch # 计算平均损失
            epoch_accs = running_accs / dataset_sizes[phase] # 计算平均准确率

            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch) # 记录训练损失
                writer.add_scalar('Accuracy/train', epoch_accs, epoch) # 记录训练准确率

                wandb.log({'Loss': epoch_loss,'Accuracy': epoch_accs})
                #wandb.log({'Accuracy': epoch_accs})

            else:
                writer.add_scalar('Loss/val', epoch_loss, epoch) # 记录验证损失
                writer.add_scalar('Accuracy/val', epoch_accs, epoch) # 记录验证准确率

                wandb.log({'Val_Loss': epoch_loss,'Val_Accuracy': epoch_accs})
                #wandb.log({'Val_Accuracy': epoch_accs})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accs)) # 打印损失和准确
            if phase == 'train':
                step_lr_scheduler.step() # 学习率调度器

    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch) # 记录模型参数分布
    writer.close() # 关闭SummaryWriter
    wandb.finish() # 结束wandb记录
    return model

if __name__ == '__main__':
    image_size = 64 # 图像统一缩放大小
    crop_size = 48 # 图像裁剪大小,即最后输入模型的图像大小
    nclass = 2 # 表情类别数

    model = CVFacialExpressionModel(nclass) # 实例化模型

    data_dir = './source_data/txt' # 数据集路径
    data_type = ['train', 'val']

    if not os.path.exists('models'):
        os.makedirs('models') # 创建模型保存路径

    use_gpu = torch.cuda.is_available() # 判断是否有GPU
    if use_gpu:
        print('use gpu===============================================')
        model = model.cuda() # 将模型放到GPU上
    print(model)

    wandb.watch(model, log='all', log_graph=True)  # 监控模型训练过程

    ## TODO 预处理
    # 数据预处理 - 数据增强
    data_transformers = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(crop_size), # 随机裁剪
            transforms.RandomHorizontalFlip(), # 随机水平翻转
            transforms.RandomVerticalFlip(), # 随机垂直翻转
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size), # 调整大小
            transforms.CenterCrop(crop_size), # 中心裁剪
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化
        ]),
    }

    ##  dataset
    image_datasets = {x: MouthImageDataset(os.path.join(data_dir, x + '.txt'), data_transformers.get(x)) for x in data_type}
    ##  dataloader
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in data_type}

    dataset_sizes = {x: len(image_datasets[x]) for x in data_type} # 数据集大小


    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss() #交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #随机梯度下降优化器 参数说明 ：model.parameters()表示需要优化的参数，lr表示学习率，momentum表示动量
    step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #学习率调度器，每隔10个epoch将学习率乘以0.1

    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=5) # 训练模型

    torch.save(model.state_dict(), 'models/model.pt') # 保存模型
