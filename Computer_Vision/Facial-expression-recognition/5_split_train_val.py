# 数据集拆分  训练集和验证集
import random

def split_train_val(file_dir, train_ratio=0.9):
    fall = open(file_dir)
    fileids = file_dir.split('.')
    fileid = fileids[0]
    train_file = open(fileid + '_train.txt', 'w')
    val_file = open(fileid + '_val.txt', 'w')

    lines = fall.readlines()
    # 计算90%和10%的行数
    num_lines_train = int(train_ratio * len(lines))
    num_lines_val = len(lines) - num_lines_train

    # 随机选择90%的行
    lines_train = random.sample(lines, num_lines_train)

    # 剩余的10%的行
    lines_val = [line for line in lines if line not in lines_train]

    # 将90%的行写入第一个输出文件
    with open(fileid + '_train.txt', 'w', encoding='utf-8') as f:
        f.writelines(lines_train)

    # 将10%的行写入第二个输出文件
    with open(fileid + '_val.txt', 'w', encoding='utf-8') as f:
        f.writelines(lines_val)

    fall.close()


if __name__ == '__main__':
    split_train_val('0.txt')
    split_train_val('1.txt')
