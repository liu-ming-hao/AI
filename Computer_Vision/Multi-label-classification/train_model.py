'''
训练模型
'''

import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms

#from model_choose import ResNet18
from net import ResNet18
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from data_second_build_dataset import VOCDataset
import onnx
import onnxruntime

import tqdm

from data_first_getlabel import load_img_name_list,load_cls_labels_from_npy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_model(model,optimizer,scheduler,epochs):

    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        for phase in ['train','val']:
            gt_labels = []
            pred_labels = []
            if phase == 'train':
                model.train() # 设置为训练模式
            else:
                model.eval() # 设置为验证模式
            running_loss = 0.0
            num_batch = 0

            for data in tqdm.tqdm(dataloaders[phase]):
                inputs,labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.multilabel_soft_margin_loss(outputs,labels)

                for i in range(0,labels.shape[0]):
                    gt_labels.append(labels[i].cpu().detach().numpy())
                    pred_labels.append(outputs[i].cpu().detach().numpy())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                num_batch += 1

            epoch_loss = running_loss / num_batch
            gts = np.array(gt_labels)
            preds = np.array(pred_labels)

            accs = get_acc(gts,preds)
            epoch_acc = np.mean(accs) if len(accs) > 0 else 0.0

            # 保存训练和验证的准确率和损失
            if phase == 'train':
                train_accs.append(epoch_acc)
                train_losses.append(epoch_loss)

                # 保存到tensorboard 
                writer.add_scalar('train_loss',epoch_loss,epoch)
                writer.add_scalar('train_acc',epoch_acc,epoch)

                scheduler.step()
            else:
                val_accs.append(epoch_acc)
                val_losses.append(epoch_loss)

                # 保存到tensorboard
                writer.add_scalar('val_loss',epoch_loss,epoch)
                writer.add_scalar('val_acc',epoch_acc,epoch)

            # 打印训练和验证的准确率和损失
            print(f"Phase: {phase}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch) # 记录模型参数分布
    writer.close()


    train_accs = np.array(train_accs, dtype=np.float32).reshape(-1)
    val_accs = np.array(val_accs, dtype=np.float32).reshape(-1)
    train_losses = np.array(train_losses, dtype=np.float32).reshape(-1)
    val_losses = np.array(val_losses, dtype=np.float32).reshape(-1)
    
    # 创建x轴数据点
    x = np.arange(len(train_accs))
    
    # 保存数据
    np.savetxt('train_accs.txt', train_accs)
    np.savetxt('val_accs.txt', val_accs)
    np.savetxt('train_losses.txt', train_losses)
    np.savetxt('val_losses.txt', val_losses)
    # try:
    #     # 确保数据是一维数组
    #     train_accs = np.array(train_accs, dtype=np.float32).reshape(-1)
    #     val_accs = np.array(val_accs, dtype=np.float32).reshape(-1)
    #     train_losses = np.array(train_losses, dtype=np.float32).reshape(-1)
    #     val_losses = np.array(val_losses, dtype=np.float32).reshape(-1)
        
    #     # 创建x轴数据点
    #     x = np.arange(len(train_accs))
        
    #     # 保存数据
    #     np.savetxt('train_accs.txt', train_accs)
    #     np.savetxt('val_accs.txt', val_accs)
    #     np.savetxt('train_losses.txt', train_losses)
    #     np.savetxt('val_losses.txt', val_losses)

    #     # 使用最基础的绘图方式
    #     fig = plt.figure(figsize=(12, 6))
        
    #     # 绘制所有曲线在同一个图上
    #     plt.plot(x, train_accs, 'b-', label='train_acc')
    #     plt.plot(x, val_accs, 'g-', label='val_acc')
    #     plt.plot(x, train_losses, 'r-', label='train_loss')
    #     plt.plot(x, val_losses, 'y-', label='val_loss')
        
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Value')
    #     plt.title('Training and Validation Metrics')
    #     plt.legend()
    #     plt.grid(True)
        
    #     # 保存图形
    #     plt.savefig('train_val_acc_loss.png', dpi=300, bbox_inches='tight')
    #     plt.close()
        
    # except Exception as e:
    #     print(f"Error during plotting: {str(e)}")
    #     print(f"Data shapes after reshape:")
    #     print(f"train_accs: {train_accs.shape}, values: {train_accs}")
    #     print(f"val_accs: {val_accs.shape}, values: {val_accs}")

    return model


def get_acc(gts, preds):
    """
    计算多标签分类的平均精度(Average Precision)
    
    Args:
        gts: numpy array, shape (N, C), N是样本数量,C是类别数量
             真实标签,每个元素是0或1,表示样本是否属于该类别
        preds: numpy array, shape (N, C)
              预测的概率值,每个元素是0-1之间的浮点数
              
    Returns:
        list: 每个类别的平均精度(AP)列表
        
    Example:
        >>> gts = np.array([[1,0,1],
                           [0,1,1],
                           [1,0,0]])
        >>> preds = np.array([[0.9,0.1,0.8],
                            [0.2,0.7,0.9], 
                            [0.8,0.3,0.1]])
        >>> aps = get_acc(gts, preds)
        >>> print(aps)
        [0.833, 1.0, 0.75] # 三个类别的AP值
    """
    avg_precisions = []

    # 对每个类别分别计算AP
    for i in range(gts.shape[1]):
        gt = gts[:,i]  # 第i个类别的真实标签
        pred = preds[:,i]  # 第i个类别的预测概率
        
        # 按预测概率从大到小排序
        sort_idx = np.argsort(pred)[::-1]
        
        # 计算真阳性(TP)和假阳性(FP)
        # 根据预测概率从大到小的顺序重排真实标签(gt),并判断每个位置是为1(正样本)
        # gt[sort_idx]按预测概率排序后的真实标签
        # == 1判断是否为正样本
        # 返回一个布尔数组,True表示该位置是真正例(预测概率较大且实际为正样本)
        true_positives = gt[sort_idx] == 1  # 预测正确的正样本
        false_positives = gt[sort_idx] == 0  # 预测错误的负样本
        
        # 计算累积TP和FP数量
        true_positives_count = np.cumsum(true_positives)
        false_positives_count = np.cumsum(false_positives)

        # 计算召回率 recall = TP / (所有正样本数)
        recall = true_positives_count / np.sum(gt > 0)

        # 避免除0错误
        epsilon = 1e-10
        
        # 计算精确率 precision = TP / (TP + FP)
        positives = true_positives_count + false_positives_count
        precision = true_positives_count / (positives + (positives == 0) * epsilon)

        # 计算平均精度(AP)
        # 在11个召回率阈值(0,0.1,...,1.0)上计算精确率的平均值
        avg_precision = 0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall_threshold = precision[recall >= t]
            if precisions_at_recall_threshold.size > 0:
                max_precision = np.max(precisions_at_recall_threshold)
            else:
                max_precision = 0
            avg_precision += max_precision / 11
            
        avg_precisions.append(avg_precision)

    return avg_precisions


if __name__ == '__main__':
    image_size = 256
    crop_size = 224
    num_classes = 20
    batch_size = 32
    epochs = 50

    model = ResNet18(num_classes)

    if not os.path.exists('models'):
        os.makedirs('models')

    if torch.cuda.is_available():
        print('CUDA enabled=================================')
        model = model.cuda()

    # 输出模型形状到tensorboard
        # 创建带时间戳的日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', f'run_{timestamp}')
    writer = SummaryWriter(log_dir)
    dummy_input = torch.randn(1, 3, crop_size, crop_size)  # 创建一个示例输入
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    writer.add_graph(model, dummy_input)  # 添加模型结构图到tensorboard



    # 数据增强
    data_transforms = {'train':transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ]),
        'val':transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])}

    # 加载数据集
    cls_labels_dict = np.load('cls_labels.npy',allow_pickle=True).item()
    train_list_path = 'D:\pythoncode\muke\\17\multi_label\multi_label\VOC2012\ImageSets\Main\\train.txt'
    val_list_path = 'D:\pythoncode\muke\\17\multi_label\multi_label\VOC2012\ImageSets\Main\\val.txt'
    dataroot = 'D:\pythoncode\muke\\17\multi_label\multi_label\VOC2012'
    train_img_name_list = load_img_name_list(train_list_path)
    val_img_name_list = load_img_name_list(val_list_path)
    train_label_list = load_cls_labels_from_npy(cls_labels_dict,train_img_name_list)
    val_label_list = load_cls_labels_from_npy(cls_labels_dict,val_img_name_list)
    # 构建数据集
    train_dataset = VOCDataset(train_img_name_list,train_label_list,dataroot,data_transform =data_transforms['train'])
    val_dataset = VOCDataset(val_img_name_list,val_label_list,dataroot,data_transform =data_transforms['val'])
    # 构建数据加载器
    dataloaders = {'train':DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4),
                   'val':DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=4)}

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1) # 学习率衰减,每7个epoch衰减为原来的10%

    train_model(model,optimizer,scheduler,epochs)





    # 保存模型
    torch.save(model.state_dict(),'models/model.pth')

    # 保存模型为onnx格式
    torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True)

    

