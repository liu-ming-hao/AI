# 多标签分类项目 (Multi-label Classification)

## 项目简介
本项目实现了基于ResNet18的多标签图像分类系统，使用PASCAL VOC 2012数据集进行训练。该系统可以同时识别图像中存在的多个对象类别。

## 项目结构 

## 核心功能

### 1. 数据处理
- 支持PASCAL VOC 2012数据集的处理和加载
- 实现了数据增强:
  - 随机裁剪
  - 水平翻转
  - 随机旋转
  - 颜色抖动
- 训练集和验证集的自动划分

### 2. 模型架构
- 基于ResNet18的深度学习模型
- 修改了最后的全连接层以支持多标签输出
- 使用多标签软边界损失函数(Multilabel Soft Margin Loss)

### 3. 训练特性
- 支持GPU加速训练
- 使用Adam优化器
- 实现了学习率调度(每7个epoch衰减为原来的10%)
- 使用TensorBoard记录训练过程:
  - 损失值变化
  - 准确率变化
  - 模型参数分布
- 自动保存训练指标和模型权重

### 4. 评估指标
- 平均精度(Average Precision, AP)计算
- 支持多类别的精确率-召回率评估
- 训练过程中同时监控训练集和验证集性能

### 5. 预测功能
- 支持批量图像预测
- 可视化预测结果:
  - 显示前三个最可能的类别
  - 显示每个类别的置信度
- 支持预测结果的保存和实时显示

### 预测图像

## 环境要求
- Python 3.6+
- PyTorch
- torchvision
- OpenCV
- PIL
- numpy
- matplotlib
- tensorboard

## 使用说明

### 训练模型 

预测结果将保存在`result`目录下。

## 模型性能
- 训练损失和准确率记录在`train_losses.txt`和`train_accs.txt`
- 验证损失和准确率记录在`val_losses.txt`和`val_accs.txt`
- 可通过TensorBoard查看详细的训练过程： 

## 注意事项
- 确保数据集路径正确配置
- 预测前请确保模型权重文件存在于`models`目录下
- 建议使用GPU进行训练以获得更好的性能

## 许可证
MIT License 