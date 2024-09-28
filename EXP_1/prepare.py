import os
import numpy as np
from mindspore import Tensor, context, Model, load_checkpoint, load_param_into_net
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype

# 设置MindSpore的运行模式为图模式
context.set_context(mode=context.GRAPH_MODE)


# 定义LeNet5网络结构
class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 初始化卷积层和全连接层，设置权重初始化
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # 网络的前向传播定义
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    # 创建数据集


def create_dataset(data_path, batch_size=1, repeat_size=1, num_parallel_workers=1):
    # 加载MNIST数据集
    mnist_ds = ds.MnistDataset(data_path)
    # 定义数据预处理操作
    resize_op = CV.Resize((32, 32), interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(1.0 / 255.0, 0.0)
    rescale_nml_op = CV.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 应用预处理操作
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label")
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image")

    # 数据混洗和批处理
    mnist_ds = mnist_ds.shuffle(10000)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds


# 主程序
if __name__ == '__main__':
    # 初始化MindSpore运行上下文
    context.set_context(mode=context.GRAPH_MODE)

    # 实例化网络、损失函数和优化器
    net = LeNet5()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # 配置检查点保存
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)


    # 定义训练和测试函数
    def test_net(model, data_path):
        # 评估模型
        ds_eval = create_dataset(os.path.join(data_path, "test"))
        acc = model.eval(ds_eval, dataset_sink_mode=False)
        print("Accuracy: {}".format(acc))


    def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
        # 训练模型
        ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
        model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)

        # 训练参数和数据路径


    train_epoch = 1
    mnist_path = "./datasets/MNIST_Data/"
    repeat_size = 1

    # 实例化Model
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": nn.Accuracy()})

    # 执行训练
    train_net(model, train_epoch, mnist_path, repeat_size, ckpoint, False)