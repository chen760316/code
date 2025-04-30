"""
获取LIME选出的特征：从LIME的解释结果中获取模型对某个预测的解释，并找出LIME选出的特征。
计算稀疏性：通过计算LIME解释中被选中的特征数量与所有特征数量之间的比例来得出稀疏性。
"""

import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use('TkAgg')  # 或者尝试 'Agg'，它是非交互式的
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import re

def extract_feature_name(feature):
    """
    根据特征的离散化标签，提取原始的特征名。
    支持单一通配符和双通配符的标签形式。
    """
    # 正则表达式匹配单个通配符或者两个通配符的情况
    match_single = re.match(r'([a-zA-Z0-9_]+)\s*(<=|<|>=|>|==)\s*(\d+(\.\d+)?)', feature)
    match_double = re.match(r'(\d+(\.\d+)?)\s*(<=|<|>=|>|==)\s*([a-zA-Z0-9_]+)\s*(<=|<|>=|>|==)\s*(\d+(\.\d+)?)', feature)

    if match_single:
        # 只有一个通配符，提取通配符左侧的特征名
        return match_single.group(1)

    elif match_double:
        # 两个通配符，提取通配符中间的特征名
        return match_double.group(4)

    # 如果都不是，返回原始特征名（原始输入没有离散化）
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

# 设定参数
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# 加载数据
data = pd.read_csv('../datasets/real_outlier/Cardiotocography.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# 使用XGBoost训练模型
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=random_state)

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

# 使用LIME解释模型
# 初始化LIME解释器
# 获取特征名
feature_names = data.columns[:-1]  # 获取特征列名，不包括最后一列标签
explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode="classification", discretize_continuous=True, feature_names=feature_names)

# 选择一个预测为离群值的样本（通常我们选择预测为1的离群样本）
idx = np.where(y_pred == 1)[0][0]  # 选择第一个预测为离群样本的索引
print(f"Explaining prediction for instance {idx}")

# 对该样本进行LIME解释
exp = explainer.explain_instance(X_test[idx], model.predict_proba)

# 显示LIME的解释结果
exp.show_in_notebook(show_table=True, show_all=False)

# 如果想保存解释为图像
exp.as_pyplot_figure()
plt.show()

# 使用表格保存lime的解释结果
# 获取LIME解释的特征重要性列表
feature_list = exp.as_list()

# 将解释结果转换为DataFrame
features = [f[0] for f in feature_list]
weights = [f[1] for f in feature_list]

df = pd.DataFrame({
    'Feature': features,
    'Weight': weights
})

# 打印表格
print(df)

# 保存为CSV
df.to_csv('lime_explanation.csv', index=False)

# 或者保存为Excel
df.to_excel('lime_explanation.xlsx', index=False)

"""
计算稀疏性：
稀疏性（Sparsity） 衡量了LIME解释中使用的特征数量相对于模型特征总数的比例。通过计算这一比例，您可以量化解释的简洁程度，并进一步研究用户对于不同稀疏性水平的接受度。
"""

X_instance = X_test[idx]

# 计算忠实度
sparsity_score = sparsity(explainer, model, X_instance)  # 选择X_test的一个样本进行测试
print(f"Sparsity score: {sparsity_score:.4f}")