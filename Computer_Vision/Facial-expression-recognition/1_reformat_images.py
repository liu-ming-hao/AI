import os
import cv2

# 数据预处理一： 格式统一转换成jpg,同时删除opencv无法读取的文件
def list_files(FilePath):
    list_files = os.walk(FilePath)  # 得到文件夹下所有文件及文件夹
    for root, dirs, files in list_files:   # 遍历所有文件及文件夹
        for d in dirs:  # 遍历所有文件夹
            print('root+d:' + os.path.join(root, d))  # 输出文件夹路径
        for f in files:  # 遍历所有文件
            fileid = f.split('.')[0]  # 得到文件名
            filepath = os.path.join(root,f)  # 得到文件路径
            print('root+f:' + filepath)
            try:
                src = cv2.imread(filepath, 1) #彩色模式读取图像
                os.remove(filepath)
                cv2.imwrite(os.path.join(root, fileid + '.jpg'), src)
            except:
                os.remove(filepath) # 删除无法读取的图
                continue

if __name__ == '__main__':
    list_files('source_data')
