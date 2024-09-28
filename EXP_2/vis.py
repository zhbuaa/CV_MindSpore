# 导入必要的库  
import os
import math
import stat
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 设置matplotlib的后端，以便在Tkinter窗口中显示图像
import matplotlib.pyplot as plt

# 导入MindSpore相关库  
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common.initializer import Normal
from mindspore import dtype as mstype
from mindspore.train.callback import TimeMonitor, Callback
from mindspore import Model, Tensor, context, save_checkpoint, load_checkpoint, load_param_into_net

# 假设已经有一个resnet50的模型定义在resnet.py文件中  
from resnet import resnet50

# 设置MindSpore的运行模式为图模式  
context.set_context(mode=context.GRAPH_MODE)

# 定义训练和验证数据集的路径  
train_data_path = './data/Canidae/train'
val_data_path = './data/Canidae/val'


def create_dataset(data_path, batch_size=24, repeat_num=1, training=True):
    """  
    定义数据集加载和预处理函数  
    :param data_path: 数据集路径  
    :param batch_size: 批处理大小  
    :param repeat_num: 数据集重复次数  
    :param training: 是否为训练集，影响数据增强策略  
    :return: 预处理后的数据集  
    """
    # 加载数据集  
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True)

    # 设置图像大小和归一化参数  
    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # 根据是否为训练集选择不同的数据增强策略  
    if training:
        trans = [
            CV.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),  # 随机裁剪并解码调整大小  
            CV.RandomHorizontalFlip(prob=0.5),  # 随机水平翻转  
            CV.Normalize(mean=mean, std=std),  # 归一化  
            CV.HWC2CHW()  # 转换图像数据从HWC到CHW  
        ]
    else:
        trans = [
            CV.Decode(),  # 解码图像  
            CV.Resize(256),  # 调整图像大小  
            CV.CenterCrop(image_size),  # 中心裁剪  
            CV.HWC2CHW()  # 转换图像数据从HWC到CHW  
        ]

        # 将标签数据类型转换为int32
    type_cast_op = C.TypeCast(mstype.int32)

    # 应用数据转换和批量处理  
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set


# 创建训练数据集
train_ds = create_dataset(train_data_path)

# 获取一批数据用于展示  
data = next(train_ds.create_dict_iterator())
images = data["image"]
labels = data["label"]

# 打印图像和标签的形状  
print("Tensor of image", images.shape)
print("Labels:", labels)

# 定义类别名称  
class_name = {0: "dogs", 1: "wolves"}
count = 1  # 用于控制子图的位置  

# 展示图像  
plt.figure(figsize=(12, 5))
for i in images:
    plt.subplot(3, 8, count)  # 假设每行3张图，总共8行（这里可能需要根据实际batch_size调整）  
    picture_show = np.transpose(i.asnumpy(), (1, 2, 0))  # 将CHW转换为HWC  
    picture_show = picture_show / np.amax(picture_show)  # 归一化到0-1  
    picture_show = np.clip(picture_show, 0, 1)  # 裁剪到0-1范围  
    plt.imshow(picture_show)  # 显示图像  
    plt.title(class_name[int(labels[count - 1].asnumpy())])  # 显示图像标题  
    plt.xticks([])  # 不显示x轴刻度  
    plt.axis("off")  # 不显示坐标轴  
    count += 1
plt.show()  # 显示图像