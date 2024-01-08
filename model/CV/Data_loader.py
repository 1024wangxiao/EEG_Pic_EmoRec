import cv2
import random
import numpy as np
import os

# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img

# 定义训练集数据读取器
def data_loader(datadir, batch_size=20, mode = 'train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            #依次读取每张图片的名称首字母用于标记标签
            # 7类表情 'sad', 'disgust', 'happy', 'fear', 'surprise', 'neutral', 'angry'
            # SA开头表示sad表情用0标签
            # DI开头表示disgust表情用1标签
            # HA开头表示happy表情用2标签，以此类推
            if name[:2] == 'SA':
                label = 0
            elif name[:2] == 'DI':
                label = 1
            elif name[:2] == 'HA':
                label = 2
            elif name[:2] == 'FE':
                label = 3
            elif name[:2] == 'SU':
                label = 4
            elif name[:2] == 'NE':
                label = 5
            elif name[:2] == 'AN':
                label = 6
            else:
                raise('Not excepted file name')
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).reshape(-1, 1)
            yield imgs_array, labels_array
    return reader

# 定义验证集数据读取器
def valid_data_loader(datadir, batch_size=20, mode='valid'):
    filenames = os.listdir(datadir)
    def reader():
        batch_imgs = []
        batch_labels = []
            # 根据图片文件名加载图片，并对图像数据作预处理
        for name in filenames:
            filepath = os.path.join(datadir, name)
            # 每读取一个样本的数据，就将其放入数据列表中
            img = cv2.imread(filepath)
            img = transform_img(img)
            #根据名称判断标签
            if name[:2] == 'SA':
                label = 0
            elif name[:2] == 'DI':
                label = 1
            elif name[:2] == 'HA':
                label = 2
            elif name[:2] == 'FE':
                label = 3
            elif name[:2] == 'SU':
                label = 4
            elif name[:2] == 'NE':
                label = 5
            elif name[:2] == 'AN':
                label = 6
            else:
                raise('Not excepted file name')
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).reshape(-1, 1)
            yield imgs_array, labels_array

    return reader