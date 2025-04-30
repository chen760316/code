"""
特征重要性：使用 xgb.plot_importance(model, importance_type='weight', max_num_features=10) 显示了特征的重要性。
importance_type 可以是 weight（特征在树中出现的次数），gain（特征带来的信息增益）或者 cover（特征覆盖的样本数）。通过这个方法，你可以看到各个特征在 SVM 模型中的贡献。
树结构：通过 model.get_booster().get_dump() 获取模型的树结构。此方法将返回模型的所有树的文本描述。你可以查看每棵树的结构，理解模型如何根据不同的特征分割数据。
树的可视化：通过 xgb.plot_tree(model, num_trees=0) 可视化 SVM 中的决策树结构。num_trees=0 指的是第一棵树。如果想查看其他树，可以改变 num_trees 参数。

解释机制说明：
    特征重要性：plot_importance 展示了各特征对模型的贡献程度。你可以选择不同的 importance_type 来定义如何衡量特征的重要性。
    树的结构：XGBoost 使用基于树的模型来做出预测，你可以通过查看每棵树的结构，了解决策是如何基于不同的特征值进行划分的。这对于理解模型的决策过程非常有帮助。
"""

import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('TkAgg')  # 或者尝试 'Agg'，它是非交互式的
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 加载数据
data = pd.read_csv('../datasets/real_outlier/annthyroid.csv')  # 请根据实际路径修改
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 划分训练集和测试集
random_state = 42  # 设置随机种子，确保结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# 初始化线性SVM模型
model = SVC(kernel='linear', C=20, random_state=random_state, probability=True)

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

# 1. 支持向量（Support Vectors）
support_vectors_idx = model.support_  # 获取支持向量的索引
print(f"Number of Support Vectors: {len(support_vectors_idx)}")
print("支持向量的索引：", support_vectors_idx)

# 显示支持向量的样本
X_support_vectors = X_train[support_vectors_idx]
y_support_vectors = y_train[support_vectors_idx]
print(f"Support Vector Samples: {X_support_vectors}")

# 2. 查看模型系数（仅适用于线性SVM）
if hasattr(model, 'coef_'):  # 确保模型使用的是线性核
    feature_weights = model.coef_[0]  # 获取特征权重（系数）
    print("\nFeature Weights (Coefficients):")
    for i, weight in enumerate(feature_weights):
        print(f"Feature {i}: Weight {weight:.4f}")

# 可视化特征的系数
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_weights)), feature_weights)
plt.xlabel('Feature Index')
plt.ylabel('Weight')
plt.title('Feature Weights (Coefficients) for Linear SVM')
plt.show()

