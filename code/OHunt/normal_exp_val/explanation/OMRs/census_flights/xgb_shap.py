"""
使用SHAP来解释模型预测：我用 shap.Explainer 创建了一个 SHAP 解释器对象，并计算了每个样本的 SHAP 值。
解释离群值样本：通过 shap_values[idx] 提取了目标样本的 SHAP 值。然后我们通过 shap.force_plot 可视化该样本的 SHAP 解释。
可视化全局特征重要性：使用 shap.summary_plot 绘制了所有样本的 SHAP 值，从而理解全局特征的重要性。

SHAP 解释：
    SHAP值：SHAP（Shapley additive explanations）为每个特征计算一个值，表示该特征对预测结果的影响。SHAP值的累加是模型输出的总和。SHAP值可以为个别预测提供精确的解释。
    force_plot：用来显示单个样本的解释，图中展示了每个特征对模型预测的贡献。
    summary_plot：展示了所有样本的SHAP值，帮助我们了解模型中各特征的全局重要性。
"""

import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import shap
import matplotlib
matplotlib.use('TkAgg')  # 或者尝试 'Agg'，它是非交互式的
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# 设定参数
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# subsection 选用的大规模数据集
# file_path = "../../../datasets/multi_class/adult.csv"
# data = pd.read_csv(file_path)

# subsection 选用的大规模数据集
file_path = "../../../datasets/multi_class/flights.csv"
data = pd.read_csv(file_path).dropna(subset=['WEATHER_DELAY'])

# 如果数据量超过20000行，就随机采样到20000行
if len(data) > 10000:
    data = data.sample(n=10000, random_state=42)

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

# section 数据特征缩放和数据加噪

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# 数据标准化（对所有特征进行标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# === 2. 拆分训练 / 测试 / 验证 ===
original_indices = np.arange(len(X))
# 按照 70% 训练集，20% 验证集，10% 测试集的比例随机划分数据集
X_train, X_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)
# 从临时数据集（30%）中划分出 10% 测试集和 20% 验证集
X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(X_temp, y_temp, temp_indices, test_size=0.33, random_state=1)


# === 3. 训练 XGBoost 模型 ===
# 定义超参数搜索空间
param_grid = {
    'n_estimators': [50, 100, 150],      # 树的数量
    'max_depth': [3, 6, 9],              # 树的最大深度
    'learning_rate': [0.01, 0.1, 0.2],   # 学习率
    'subsample': [0.8, 1.0],             # 子样本比例
    'colsample_bytree': [0.8, 1.0],      # 每棵树的特征采样比例
    'scale_pos_weight': [1, 2],          # 类别不平衡时调节
    'class_weight': [None]               # 兼容原格式，XGBoost 实际不支持 class_weight（占位）
}

# 初始化 XGBoost 模型（不用指定 n_estimators 之类的，这些会由 GridSearch 决定）
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    booster='gbtree',
    random_state=42,
    verbosity=0
)

# 初始化 GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy',
                           cv=2,
                           verbose=2,
                           n_jobs=-1)

# 拟合模型
grid_search.fit(X_val, y_val)

# 输出最佳超参数和交叉验证得分
print("最佳超参数组合：", grid_search.best_params_)
print("最佳交叉验证准确度：", grid_search.best_score_)

# 获取最佳模型
model = grid_search.best_estimator_

# 训练模型
model.fit(X_train, y_train)

# 预测结果
y_pred = model.predict(X_test)

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

# 计算Precision-Recall曲线
precision_curve, recall_curve, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
pr_auc = auc(recall_curve, precision_curve)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# 可视化Precision-Recall曲线
plt.figure(figsize=(6, 6))
plt.plot(recall_curve, precision_curve, color='b', label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# 选择一个预测为离群值的样本（通常我们选择预测为1的离群样本）
idx = np.where(y_pred == 1)[0][0]  # 选择第一个预测为离群样本的索引
print(f"Explaining prediction for instance {idx}")

# 使用SHAP进行解释
# 创建SHAP解释器
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 获取该样本的SHAP值
"""正向并且绝对值更大的沙普利值更能体现出该特征导致XGBoost预测值为1"""
shap_value_for_instance = shap_values[idx]
print(f"SHAP values for instance {idx}: {shap_value_for_instance.values}")

# 可视化该实例的SHAP值
shap.initjs()  # 初始化JS可视化工具
shap.force_plot(shap_values[idx].base_values, shap_values[idx].values, X_test.iloc[idx], feature_names=data.columns[:-1])

# 如果需要查看所有样本的SHAP值，可使用summary_plot
shap.summary_plot(shap_values.values, X_test, feature_names=data.columns[:-1])

# 保存SHAP图像
shap.force_plot(shap_values[idx].base_values, shap_values[idx].values, X_test.iloc[idx], feature_names=data.columns[:-1])
plt.savefig('shap_explanation.png')  # 保存为图片


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
    计算SHAP解释器的稀疏性（Sparsity）。

    explainer: SHAP的解释器对象
    X_instance: 要解释的单个样本（特征）
    """
    # 1. 使用SHAP解释器解释该样本
    shap_values = explainer.shap_values(X_instance)  # 返回SHAP值

    # 2. 获取SHAP选出的特征
    # 如果是二分类问题，通常会有两个类的SHAP值，我们选择模型预测的类对应的SHAP值
    if isinstance(shap_values, list):
        # 对于二分类，shap_values 会是一个列表
        shap_values = shap_values[1]  # 选择第二个类的shap值，通常是正类

    # 将SHAP值大于0的特征作为重要特征
    selected_features = [feature for feature, value in zip(X_instance.index, shap_values) if abs(value) > 0.1]

    # 3. 计算稀疏性：选定特征的数量与总特征数量的比例
    total_features = len(X_instance.index)
    selected_features_count = len(selected_features)
    sparsity_score = selected_features_count / total_features
    return sparsity_score


def fidelity(model, explainer, X_instance, num_features=10):
    """
    计算SHAP解释器的忠实度（Fidelity）。

    model: 已训练的XGBoost模型
    explainer: SHAP的解释器对象
    X_instance: 要解释的单个样本（特征）
    num_features: SHAP解释时使用的特征数量
    """

    # 1. 获取模型的真实预测结果
    # 确保 X_instance 是 numpy 数组类型
    X_instance = np.array(X_instance)  # 转换为 numpy 数组
    true_prediction = model.predict_proba(X_instance.reshape(1, -1))[:, 1]

    # 2. 使用SHAP解释器解释该样本
    shap_values = explainer.shap_values(X_instance)  # 返回SHAP值

    # 如果是二分类问题，shap_values 会是一个列表，我们选择第二类的SHAP值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 选择正类的SHAP值

    # 3. 获取SHAP选出的特征（选取绝对值较大的特征）
    selected_features = np.argsort(np.abs(shap_values))[-num_features:]

    # 4. 构建基于SHAP选出的特征的修改过的输入
    modified_input = np.zeros_like(X_instance)  # 其他特征设为0

    # 还原SHAP选出的特征
    for idx in selected_features:
        modified_input[idx] = X_instance[idx]

    # 5. 使用修改后的输入重新进行预测
    modified_prediction = model.predict_proba(modified_input.reshape(1, -1))[:, 1]

    # 6. 计算忠实度：我们用均方误差（MSE）来衡量真实预测和修改后预测之间的差异
    fidelity_score = mean_squared_error(true_prediction, modified_prediction)

    return fidelity_score


def fidelity_all(model, explainer, X_instance, num_features=10):
    """
    计算SHAP解释器的忠实度（Fidelity）。

    model: 已训练的XGBoost模型
    explainer: SHAP的解释器对象
    X_instance: 要解释的单个样本（特征）
    num_features: SHAP解释时使用的特征数量
    """

    # 1. 获取模型的真实预测结果
    # 确保 X_instance 是 numpy 数组类型
    X_instance = np.array(X_instance)  # 转换为 numpy 数组
    true_prediction = model.predict_proba(X_instance.reshape(1, -1))[:, 1]

    # 2. 使用SHAP解释器解释该样本
    shap_values = explainer.shap_values(X_instance)  # 返回SHAP值

    # 如果是二分类问题，shap_values 会是一个列表，我们选择第二类的SHAP值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 选择正类的SHAP值

    # 3. 获取SHAP选出的特征（选取绝对值较大的特征）
    selected_features = np.argsort(np.abs(shap_values))[-num_features:]

    # 4. 构建基于SHAP选出的特征的修改过的输入
    modified_input = np.zeros_like(X_instance)  # 其他特征设为0

    # 还原SHAP选出的特征
    for idx in selected_features:
        modified_input[idx] = X_instance[idx]

    # 5. 使用修改后的输入重新进行预测
    modified_proba = model.predict_proba(modified_input.reshape(1, -1))[:, 1]

    # 获取原始预测的类别（1表示正常，0表示异常）
    initial_label = np.argmax(true_prediction)

    # 获取修改后输入的预测类别
    modified_prediction = np.argmax(modified_proba)

    # 6. 计算忠实度：比较初始预测与修改后输入的预测
    if initial_label == modified_prediction:
        return 1  # 如果预测一致，忠实度为1
    else:
        return 0  # 如果预测不一致，忠实度为0

count = 0
sparsity_scores = 0
for idx in outlier_indices:
    X_instance = X_test.iloc[idx]

    # 计算稀疏度
    sparsity_score = sparsity(explainer, model, X_instance)
    print(f"sparsity Score for instance {idx}: {sparsity_score:.4f}")
    sparsity_scores += sparsity_score

    # 计算忠实度
    fidelity_score = fidelity(model, explainer, X_instance, num_features=10)
    print(f"Fidelity Score for instance {idx}: {fidelity_score:.4f}")
    fidelity_count = fidelity_all(model, explainer, X_instance, num_features=10)
    count += fidelity_count
print("平均稀疏性得分为：", sparsity_scores/len(outlier_indices))
print("总的忠实度得分为：", count/len(outlier_indices))