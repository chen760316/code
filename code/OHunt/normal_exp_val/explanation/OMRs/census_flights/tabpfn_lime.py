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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# subsection 选用的大规模数据集
file_path = "../../../datasets/multi_class/adult.csv"
data = pd.read_csv(file_path)

# subsection 选用的大规模数据集
# file_path = "../../../datasets/multi_class/flights.csv"
# data = pd.read_csv(file_path).dropna(axis=1, how='all')

# 如果数据量超过20000行，就随机采样到20000行
if len(data) > 1000:
    data = data.sample(n=1000, random_state=42)

enc = LabelEncoder()
label_name = data.columns[-1]

# 原始数据集D对应的Dataframe
data[label_name] = enc.fit_transform(data[label_name])

# 检测非数值列
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# 为每个非数值列创建一个 LabelEncoder 实例
encoders = {}
for column in non_numeric_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码

data = data.fillna(data.mean())  # 例如，用均值填充

# section 数据特征缩放和数据加噪

X = data.values[:, :-1]
y = data.values[:, -1]
# 数据标准化（对所有特征进行标准化）
X = StandardScaler().fit_transform(X)

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

from sklearn.metrics import mean_squared_error
import re

outlier_indices = np.where(y_pred == 1)[0][:10]  # 前10个预测为离群值的样本

def extract_feature_name(feature):
    """
    根据特征的离散化标签，提取原始的特征名。
    支持单一通配符和双通配符的标签形式，包括负数。
    """
    # 支持负数（-?）
    match_single = re.match(r'([a-zA-Z0-9_]+)\s*(<=|<|>=|>|==)\s*(-?\d+(\.\d+)?)', feature)
    match_double = re.match(r'(-?\d+(\.\d+)?)\s*(<=|<|>=|>|==)\s*([a-zA-Z0-9_]+)\s*(<=|<|>=|>|==)\s*(-?\d+(\.\d+)?)', feature)

    if match_single:
        return match_single.group(1)
    elif match_double:
        return match_double.group(4)

    return feature


def sparsity(explainer, model, X_instance):
    """
    计算LIME解释器的稀疏性（Sparsity）。

    explainer: LIME的解释器对象
    X_instance: 要解释的单个样本（特征）
    num_features: LIME解释时使用的特征数量
    """
    # 1. 使用LIME解释器解释该样本
    exp = explainer.explain_instance(X_instance, model.predict_proba)
    # 2. 获取LIME选出的特征
    selected_features = [f[0] for f in exp.as_list()]
    # 3. 计算稀疏性：选定特征的数量与总特征数量的比例
    total_features = len(exp.domain_mapper.feature_names)
    selected_features_count = len(selected_features)
    sparsity_score = selected_features_count / total_features
    return sparsity_score

def fidelity(model, explainer, X_instance, num_features=10):
    """
    计算LIME解释器的忠实度（Fidelity）。

    model: 已训练的XGBoost模型
    explainer: LIME的解释器对象
    X_instance: 要解释的单个样本（特征）
    num_features: LIME解释时使用的特征数量
    """

    # 1. 获取模型的真实预测结果
    true_prediction = model.predict_proba(X_instance.reshape(1, -1))[:, 1]

    # 2. 使用LIME解释器解释该样本
    exp = explainer.explain_instance(X_instance, model.predict_proba, num_features=num_features)

    # 3. 获取LIME选出的特征
    selected_features = [f[0] for f in exp.as_list()]

    # 4. 构建基于LIME选出的特征的修改过的输入
    modified_input = np.zeros_like(X_instance)  # 其他特征设为0

    for feature in selected_features:
        # 使用提取特征名函数处理离散化标记
        original_feature = extract_feature_name(feature)

        # 获取原始特征的索引
        feature_index = exp.domain_mapper.feature_names.index(original_feature)
        modified_input[feature_index] = X_instance[feature_index]  # 还原LIME选出的特征

    # 5. 使用修改后的输入重新进行预测
    modified_prediction = model.predict_proba(modified_input.reshape(1, -1))[:, 1]

    # 6. 计算忠实度：我们用均方误差（MSE）来衡量真实预测和修改后预测之间的差异
    fidelity_score = mean_squared_error(true_prediction, modified_prediction)

    return fidelity_score

def fidelity_all(model, explainer, X_instance, num_features=10):
    true_prediction = model.predict_proba(X_instance.reshape(1, -1))[:, 1]

    # 2. 使用LIME解释器解释该样本
    exp = explainer.explain_instance(X_instance, model.predict_proba, num_features=num_features)

    # 3. 获取LIME选出的特征
    selected_features = [f[0] for f in exp.as_list()]

    # 4. 构建基于LIME选出的特征的修改过的输入
    modified_input = np.zeros_like(X_instance)  # 其他特征设为0

    for feature in selected_features:
        # 使用提取特征名函数处理离散化标记
        original_feature = extract_feature_name(feature)

        # 获取原始特征的索引
        feature_index = exp.domain_mapper.feature_names.index(original_feature)
        modified_input[feature_index] = X_instance[feature_index]  # 还原LIME选出的特征

    # 5. 使用修改后的输入重新进行预测
    modified_proba = model.predict_proba(modified_input.reshape(1, -1))[:, 1]
    initial_label = np.argmax(true_prediction)
    modified_prediction = np.argmax(modified_proba)  # 获取LIME的预测类别

    # 6. 计算忠实度：比较初始预测与修改后输入的预测
    if initial_label == modified_prediction:
        return 1  # 如果预测一致，忠实度为1
    else:
        return 0  # 如果预测不一致，忠实度为0

count = 0
for idx in outlier_indices:
    X_instance = X_test[idx]

    # 计算稀疏度
    sparsity_score = sparsity(explainer, clf, X_instance)
    print(f"sparsity Score for instance {idx}: {sparsity_score:.4f}")

    # 计算忠实度
    fidelity_score = fidelity(clf, explainer, X_instance, num_features=10)
    print(f"Fidelity Score for instance {idx}: {fidelity_score:.4f}")
    fidelity_count = fidelity_all(clf, explainer, X_instance, num_features=10)
    count += fidelity_count
print("总的忠实度得分为：", count/len(outlier_indices))