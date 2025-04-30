"""

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 或 'Agg'，适合无图形界面的环境
from captum.attr import DeepLift
from captum.attr import visualization as viz

"""我们可以使用一个简单的全连接神经网络作为模型。这个模型会接受输入特征，并输出预测的类别"""
# 创建一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 128) # 改为对应的输入特征数
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10个类别（假设是MNIST）

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 加载数据
data = pd.read_csv('../datasets/real_outlier/annthyroid.csv').sample(100)
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 将 NumPy 数组转换为 PyTorch Tensor
input_tensor_train = torch.tensor(X_train, dtype=torch.float32)  # 转换为浮点型
input_tensor_test = torch.tensor(X_test, dtype=torch.float32)

# 将标签转换为 PyTorch Tensor（如果是二分类问题，可以使用 Long 类型）
target_tensor_train = torch.tensor(y_train, dtype=torch.long)  # 对于分类问题，标签通常为 long 类型
target_tensor_test = torch.tensor(y_test, dtype=torch.long)

"""接下来，我们初始化模型、定义损失函数和优化器，并训练模型"""
# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # 分类问题常用的损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 训练模型
epochs = 20
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清零梯度

    # 前向传播
    outputs = model(input_tensor_train)

    # 计算损失
    loss = criterion(outputs, target_tensor_train)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

"""训练完模型后，我们可以用测试集对模型进行评估，计算准确率。"""
# 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 禁用梯度计算，减少内存消耗
    outputs = model(input_tensor_test)
    _, predicted = torch.max(outputs, 1)  # 获取预测类别
    correct = (predicted == target_tensor_test).sum().item()  # 计算预测正确的数量
    accuracy = correct / target_tensor_test.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

"""使用Captum解释模型预测"""
# """使用 Captum 来解释模型的预测，我们可以使用 Integrated Gradients 或其他方法。假设我们使用 Integrated Gradients"""
# from captum.attr import IntegratedGradients
#
# # 初始化 Integrated Gradients 解释器
# ig = IntegratedGradients(model)
#
# # 选择一个测试样本来解释
# test_sample = input_tensor_test[0].unsqueeze(0)  # 选择第一个测试样本，并添加一个批量维度
#
# # 计算该测试样本的 attribution
# attributions, delta = ig.attribute(test_sample, target=target_tensor_test[0], return_convergence_delta=True)
#
# # 可视化特征重要性（将 attribution 重塑为 28x28 形状，适合图像数据）
# # 如果数据是表格数据，可以调整可视化方式
# attributions = attributions.squeeze().detach().numpy()  # 移除批量维度并转为 NumPy 数组
#
# # 可视化（假设输入是表格数据，可以直接使用条形图）
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(attributions)), attributions)
# plt.xlabel("Feature Index")
# plt.ylabel("Attribution Value")
# plt.title("Feature Attribution using Integrated Gradients")
# plt.show()
#
# """如果您使用的是图像数据，您可以将 attributions 可视化为图像（例如，28x28 像素的图像）。
# 如果是表格数据，您可以使用条形图来查看各个特征的贡献。"""
#
# torch.save(model.state_dict(), 'simple_nn_model.pth')


"""使用DeepLift解释模型预测"""
"""我们使用 Captum 提供的 DeepLift 来解释模型的预测结果。首先，我们需要创建 DeepLift 解释器对象"""
# 初始化 DeepLift 解释器
deep_lift = DeepLift(model)
# 选择第一个测试样本
test_sample = input_tensor_test[0].unsqueeze(0)  # 选择第一个测试样本并添加批次维度
target_class = target_tensor_test[0].item()  # 目标标签

# 计算该测试样本的特征重要性
attributions, delta = deep_lift.attribute(test_sample, target=target_class, return_convergence_delta=True)

# 将结果转换为 NumPy 数组，移除批次维度
attributions = attributions.squeeze().detach().numpy()

# 可视化特征重要性
plt.bar(range(len(attributions)), attributions)
plt.xlabel("Feature Index")
plt.ylabel("Attribution Value")
plt.title("Feature Attribution using DeepLift")
plt.show()