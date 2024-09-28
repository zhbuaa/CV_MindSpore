# 导入必要的库
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

# 存在一个 prepare.py 文件，其中定义了 create_dataset 和 LeNet5
from prepare import create_dataset, LeNet5

# 设置MindSpore的运行模式为图模式
context.set_context(mode=context.GRAPH_MODE)


# 定义 WithLossCell 类，用于包装网络和损失函数
class WithLossCell(nn.Cell):
    """
    包装网络与损失函数
    """
    def __init__(self, network, loss_fn):
        super(WithLossCell, self).__init__()
        self._network = network
        self._loss_fn = loss_fn

    def construct(self, data, label):
        out = self._network(data)
        return self._loss_fn(out, label)

# 定义 GradWrapWithLoss 类，用于通过损失函数求取反向梯度
class GradWrapWithLoss(nn.Cell):
    """
    通过 loss 求反向梯度
    """
    def __init__(self, network):
        super(GradWrapWithLoss, self).__init__()
        self._grad_all = ops.composite.GradOperation(get_all=True, sens_param=False)
        self._network = network

    def construct(self, inputs, labels):
        gout = self._grad_all(self._network)(inputs, labels)
        return gout[0]

# 定义 FastGradientSignMethod 类，实现 FGSM 攻击
class FastGradientSignMethod:
    """
    实现 FGSM 攻击
    """
    def __init__(self, network, eps=0.07, loss_fn=None):  # 变量初始化
        self._network = network
        self._eps = eps

        with_loss_cell = WithLossCell(self._network, loss_fn)
        self._grad_all = GradWrapWithLoss(with_loss_cell)
        self._grad_all.set_train()

    def _gradient(self, inputs, labels):  # 求取梯度
        out_grad = self._grad_all(inputs, labels)
        gradient = out_grad.asnumpy()
        gradient = np.sign(gradient)
        return gradient

    def generate(self, inputs, labels):  # 实现 FGSM
        # 生成对抗样本
        inputs_tensor = Tensor(inputs)
        labels_tensor = Tensor(labels)

        gradient = self._gradient(inputs_tensor, labels_tensor)  # 产生扰动
        perturbation = self._eps * gradient  # 生成受到扰动的图片
        adv_x = inputs + perturbation
        return adv_x

    def batch_generate(self, inputs, labels, batch_size=32):  # 对数据集进行处理
        arr_x = inputs
        arr_y = labels
        len_x = len(inputs)

        batches = int(len_x / batch_size)
        rest = len_x - batches * batch_size
        res = []
        for i in range(batches):
            x_batch = arr_x[i * batch_size: (i + 1) * batch_size]
            y_batch = arr_y[i * batch_size: (i + 1) * batch_size]
            adv_x = self.generate(x_batch, y_batch)
            res.append(adv_x)
        adv_x = np.concatenate(res, axis=0)

        return adv_x

# 加载模型、损失函数和优化器
net = LeNet5()
param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
load_param_into_net(net, param_dict)

net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

# 配置和创建模型
model = Model(net, net_loss, net_opt, metrics={"Accuracy": nn.Accuracy()})

images = []
labels = []
test_images = []
test_labels = []
predict_labels = []

train_epoch = 1
mnist_path = "./datasets/MNIST_Data/"
repeat_size = 1


ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=32).create_dict_iterator(output_numpy=True)

for data in ds_test:
    images = data['image'].astype(np.float32)
    labels = data['label']
    test_images.append(images)
    test_labels.append(labels)
    pred_labels = np.argmax(model.predict(Tensor(images)).asnumpy(), axis=1)
    predict_labels.append(pred_labels)

test_images = np.concatenate(test_images)

predict_labels = np.concatenate(predict_labels)
true_labels = np.concatenate(test_labels)

# 使用不同的epsilon值进行FGSM攻击，并评估攻击效果 
fgsm = FastGradientSignMethod(net, eps=0.0, loss_fn=net_loss)
advs = fgsm.batch_generate(test_images, true_labels, batch_size=32)

adv_predicts = model.predict(Tensor(advs)).asnumpy()
adv_predicts = np.argmax(adv_predicts, axis=1)
accuracy = np.mean(np.equal(adv_predicts, true_labels))
print(accuracy)

fgsm = FastGradientSignMethod(net, eps=0.5, loss_fn=net_loss)
advs = fgsm.batch_generate(test_images, true_labels, batch_size=32)

adv_predicts = model.predict(Tensor(advs)).asnumpy()
adv_predicts = np.argmax(adv_predicts, axis=1)
accuracy = np.mean(np.equal(adv_predicts, true_labels))
print(accuracy)

# 使用matplotlib可视化原始图像和对抗样本
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg', 'Agg' 等
import matplotlib.pyplot as plt
adv_examples = np.transpose(advs[:10], [0, 2, 3, 1])
ori_examples = np.transpose(test_images[:10], [0, 2, 3, 1])

plt.figure()
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(np.squeeze(ori_examples[i]))
    plt.subplot(2, 10, i + 11)
    plt.imshow(np.squeeze(adv_examples[i]))
plt.show()
