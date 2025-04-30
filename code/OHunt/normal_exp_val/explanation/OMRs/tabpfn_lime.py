"""
TabPFNClassifier 是一个封装了多个模型（如神经网络和其他算法）的大型集成模型，可能并未直接采用 PyTorch 的模型结构。
因此，Captum 的解释方法（如 DeepLIFT）依赖于 PyTorch 特有的钩子（hooks）机制来获取模型的梯度信息。而 TabPFNClassifier 可能并没有这种机制，导致您看到 AttributeError。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
from captum.attr import DeepLift
from captum.attr import visualization as viz
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 设置 TkAgg 后端
import numpy as np

# 设定参数
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# 加载数据
file_path = "../../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../../datasets/real_outlier/annthyroid.csv"
# file_path = "../../datasets/real_outlier/optdigits.csv"
# file_path = "../../datasets/real_outlier/PageBlocks.csv"
# file_path = "../../datasets/real_outlier/pendigits.csv"
# file_path = "../../datasets/real_outlier/satellite.csv"
# file_path = "../../datasets/real_outlier/shuttle.csv"
# file_path = "../../datasets/real_outlier/yeast.csv"
data = pd.read_csv(file_path)

if len(data) > 1000:
    data = data.sample(n=1000, random_state=42)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# === 2. 拆分训练 / 测试 / 验证 ===
original_indices = np.arange(len(X))
# 按照 70% 训练集，20% 验证集，10% 测试集的比例随机划分数据集
X_train, X_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)
# 从临时数据集（30%）中划分出 10% 测试集和 20% 验证集
X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(X_temp, y_temp, temp_indices, test_size=0.33, random_state=1)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# 预测结果
y_pred = clf.predict(X_test)

# 计算性能指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# 打印各项指标
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

"""使用DeepLift解释模型预测"""

# # 转换数据为 PyTorch 张量
# input_tensor = torch.tensor(X_test[0]).unsqueeze(0)  # 第一个测试样本
# target_tensor = torch.tensor(y_test[0]).unsqueeze(0)  # 目标标签
# # 初始化 DeepLift 解释器
# dl = DeepLift(clf)
# # 计算特征重要性
# attributions, delta = dl.attribute(input_tensor, target=target_tensor, return_convergence_delta=True)
# # 打印特征重要性
# print(attributions)

"""使用shap解释模型预测"""
# # 创建 SHAP 解释器
# explainer = shap.KernelExplainer(clf.predict_proba, X_train)
# # 选择一个测试样本来解释
# shap_values = explainer.shap_values(X_test[0:1])
# # 可视化 SHAP 值
# shap.initjs()
# shap.summary_plot(shap_values, X_test)

"""使用LIME解释器"""
# 创建 LIME 解释器
explainer = LimeTabularExplainer(
    training_data=X_train,
    training_labels=y_train,
    mode="classification",
    feature_names=data.columns[:-1],
    class_names=["label"],  # 根据您的数据设置
    discretize_continuous=True
)

# 选择一个测试样本来解释
test_sample = X_test[0]

# 获取该测试样本的 LIME 解释
explanation = explainer.explain_instance(test_sample, clf.predict_proba, num_features=10)

# 获取特征贡献的列表
explanation_list = explanation.as_list()

# 提取特征名和贡献值
features = [item[0] for item in explanation_list]
contributions = [item[1] for item in explanation_list]

# 使用 matplotlib 可视化结果
plt.figure(figsize=(10, 6))
plt.barh(features, contributions, color='skyblue')
plt.xlabel('Feature Contribution')
plt.title('LIME Explanation - Feature Importance')
plt.show()