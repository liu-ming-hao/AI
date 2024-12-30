"""
==============================================
第一步： 构建多分类标签
==============================================
将20个类别转换为20个二分类标签，每个标签表示是否属于该类别。
"""

import numpy as np
import os
import argparse # 用于解析命令行参数的库


IMG_FOLDER_NAME = 'JPEGImages'
ANNOT_FOLDER_NAME = 'Annotations'

# 标签名称集合
LABEL_NAME_LIST = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
N_CLASSES = len(LABEL_NAME_LIST)

# 标签名称与标签索引的映射
LABEL_NAME_TO_INDEX = {label: index for index, label in enumerate(LABEL_NAME_LIST)}





# 读取VOC2012数据集的图片名称列表
def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    return img_name_list


# 获取图片路径
def get_img_path(img_name, voc12_root):
    """
    获取VOC数据集中图片的完整路径
    
    Args:
        img_name: 图片文件名，可以是整数格式或字符串格式
        voc12_root: VOC2012数据集的根目录路径
        
    Returns:
        str: 图片的完整路径
        
    Example:
        >>> get_img_path('2007_123456', '/path/to/VOC2012')
        '/path/to/VOC2012/JPEGImages/2007_123456'
        >>> get_img_path(2007123456, '/path/to/VOC2012') 
        '/path/to/VOC2012/JPEGImages/2007_123456'
    """
    # 如果输入的文件名不是字符串格式，则转换为VOC标准格式
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    # 拼接并返回完整的图片路径    
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name)


# 单张图片标签构建
def load_img_label_from_xml(img_name,voc12_root):
    from xml.dom import minidom
    multi_cls_label = np.zeros(N_CLASSES)
    try:
        xml_path = os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')
        if not os.path.exists(xml_path):
            print(f"Error: XML file for image {img_name} does not exist at {xml_path}")
            return np.zeros(N_CLASSES)
        elem_list = minidom.parse(xml_path).getElementsByTagName('name')
    except:
        print(f"Error: Failed to parse XML file for image {img_name}")
        return np.zeros(N_CLASSES)

    
    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in LABEL_NAME_LIST:
            cat_num = LABEL_NAME_TO_INDEX[cat_name]
            multi_cls_label[cat_num] = 1
    return multi_cls_label

# 多张图片标签构建
def load_img_label_list_from_xml(img_name_list,voc12_root):
    return [load_img_label_from_xml(img_name,voc12_root) for img_name in img_name_list]

def decode_int_filename(int_filename):
    """
    将整数格式的文件名转换为VOC数据集标准格式的文件名
    例如: 2007123456 -> 2007_123456
    
    Args:
        int_filename: 整数格式的文件名
        
    Returns:
        str: VOC标准格式的文件名字符串
    """
    s = str(int(int_filename))
    return s[:4]+'_'+s[4:]

def load_cls_labels_from_npy(cls_labels_dict,img_name_list):
    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])
 
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--voc12_root',type=str,default='D:\pythoncode\muke\\17\multi_label\multi_label\VOC2012')
    parse.add_argument('--dataset_path_train',type=str,default='D:\pythoncode\muke\\17\multi_label\multi_label\VOC2012\ImageSets\Main\\train.txt')
    parse.add_argument('--dataset_path_val',type=str,default='D:\pythoncode\muke\\17\multi_label\multi_label\VOC2012\ImageSets\Main\\val.txt')
    parse.add_argument('--out',type=str,default='cls_labels.npy')
    args = parse.parse_args()

    # 获取训练集图片名称列表
    train_img_name_list = load_img_name_list(args.dataset_path_train)
    # 获取验证集图片名称列表
    val_img_name_list = load_img_name_list(args.dataset_path_val)
    # 合并训练集和验证集图片名称列表
    img_name_list = np.concatenate([train_img_name_list, val_img_name_list],axis=0)
    # 获取标签
    cls_labels = load_img_label_list_from_xml(img_name_list,args.voc12_root)

    total_label = np.zeros(N_CLASSES)

    d = dict()

    for img_name,label in zip(img_name_list,cls_labels):
        d[img_name] = label
        total_label += label

    print(total_label)

    np.save(args.out,d)

    #读取
    cls_labels_dict = np.load(args.out,allow_pickle=True).item()
    train_img_label_list = load_cls_labels_from_npy(cls_labels_dict,train_img_name_list)
    print(train_img_label_list)






