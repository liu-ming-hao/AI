import os
# 便利文件 将文件信息存入txt文件中

def list_files(rootDir, txtfile, label=0):
    ftextfile = open(txtfile,'w')
    list_dirs = os.walk(rootDir)
    count = 0
    for root, dirs, files in list_dirs:
        for d in dirs:
            print('root+d:' + os.path.join(root, d))
        for f in files:
            print('root+f:' + os.path.join(root, f))
            ftextfile.write(os.path.join(root, f) + ' ' + str(label) + '\n')
            count += 1

    print(rootDir + ' count: ' + str(count))

if __name__ == '__main__':
    list_files('source_data/du', '0.txt', 0)
    list_files('source_data/smile', '1.txt', 1)