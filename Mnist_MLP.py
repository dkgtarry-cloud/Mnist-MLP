import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 创建一个 device 对象，用于指定模型和数据的运行设备：
# 如果检测到支持 CUDA 的 GPU，则使用 GPU；否则退回到 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理操作：
# transforms.ToTensor() 会将 PIL 图片或 numpy.ndarray 转换为 Tensor，
# 同时自动把像素值缩放到 [0,1] 范围，便于神经网络处理。
transform = transforms.ToTensor()

# 创建一个名为 train_dataset 的 MNIST 手写数字训练集对象：
# - root="./data" 指定数据存放路径
# - train=True 表示使用训练集
# - download=True 如果数据不存在会自动下载
# - transform=transform 表示对数据应用前面定义的预处理
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# 创建一个数据加载器 train_loader，用于从训练集按批次抓取数据：
# - dataset 指定要加载的数据集
# - batch_size=32 每次迭代输出 32 个样本
# - shuffle=True 打乱数据顺序，提高训练的泛化能力
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

# 定义一个多层感知机 (MLP) 模型类，继承自 nn.Module：
# 继承的好处是可以使用 Module 内置的参数管理和训练方法。
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 初始化基类，注册模型中的层和参数
        # 第一层：全连接层 (784 → 128)，输入为 28*28=784 个像素点，输出 128 个特征
        self.fc1 = nn.Linear(28*28, 128)
        # 激活函数层：ReLU，用于引入非线性，使网络能学习复杂关系
        self.relu = nn.ReLU()
        # 第二层：全连接层 (128 → 10)，将 128 维特征映射到 10 个类别（数字 0~9）
        self.fc2 = nn.Linear(128, 10)

    # 定义前向传播逻辑：描述输入数据如何流经网络各层得到输出
    def forward(self, x):
        # 展平输入：把 (batch_size, 1, 28, 28) 转换为 (batch_size, 784)
        # -1 表示自动推算 batch_size
        x = x.view(-1, 28*28)
        x = self.fc1(x)   # 通过第一层线性变换
        x = self.relu(x)  # 通过 ReLU 激活函数
        x = self.fc2(x)   # 通过第二层线性变换得到输出
        return x          # 输出 (batch_size, 10)，即每个样本在 10 个类别上的得分

# 创建模型实例，并将其放到指定设备 (CPU 或 GPU)
model = MLP().to(device)

# 定义损失函数：交叉熵损失 CrossEntropyLoss
# 适用于多分类任务，内部自动完成 softmax + log 的组合运算。
criterion = nn.CrossEntropyLoss()

# 定义优化器：随机梯度下降 (SGD)
# - model.parameters() 获取模型中需要更新的所有参数 (权重和偏置)
# - lr=0.01 设置学习率，控制参数更新的步长
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 打印模型参数所在设备，确认是否成功加载到 GPU
print("Model is on:", next(model.parameters()).device)

# 开始训练循环：共训练 5 个 epoch，每个 epoch 会完整遍历一次训练集
for epoch in range(5):
    epoch_loss = 0.0
    # 遍历数据加载器，每次取出一个 batch 的数据
    for images, labels in train_loader:
        # 把数据移动到与模型相同的设备 (CPU 或 GPU)
        images, labels = images.to(device), labels.to(device)

        # 前向传播：输入数据经过模型，得到预测结果
        outputs = model(images)
        # 计算损失：比较预测结果与真实标签的差距
        loss = criterion(outputs, labels)

        # 反向传播与参数更新
        optimizer.zero_grad()  # 清空上一次迭代的梯度，否则会累积
        loss.backward()        # 反向传播，计算每个参数的梯度
        optimizer.step()       # 使用优化器更新参数（权重 W 和偏置 b）

        epoch_loss += loss.item()  # 累加每个 batch 的损失值

    # 打印当前 epoch 的平均损失，反映整体训练效果
    print(f"Epoch [{epoch+1}/5], Avg Loss: {epoch_loss/len(train_loader):.4f}")
