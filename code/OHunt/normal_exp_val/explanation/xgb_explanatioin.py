"""
特征重要性：使用 xgb.plot_importance(model, importance_type='weight', max_num_features=10) 显示了特征的重要性。
importance_type 可以是 weight（特征在树中出现的次数），gain（特征带来的信息增益）或者 cover（特征覆盖的样本数）。通过这个方法，你可以看到各个特征在 XGBoost 模型中的贡献。
树结构：通过 model.get_booster().get_dump() 获取模型的树结构。此方法将返回模型的所有树的文本描述。你可以查看每棵树的结构，理解模型如何根据不同的特征分割数据。
树的可视化：通过 xgb.plot_tree(model, num_trees=0) 可视化 XGBoost 中的决策树结构。num_trees=0 指的是第一棵树。如果想查看其他树，可以改变 num_trees 参数。

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

# 设定参数
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# 加载数据
data = pd.read_csv('../datasets/real_outlier/annthyroid.csv')
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

# 使用XGBoost的plot_importance来展示特征的重要性
# plot_importance将展示各个特征对模型预测的重要性
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', max_num_features=10)
plt.title('Feature Importance (Weight)')
plt.show()

# 如果需要更多模型解释，也可以直接查看模型的树结构
# 使用 XGBoost 的 get_dump() 方法，可以提取模型的树结构
booster = model.get_booster()

# 获取前几棵树的结构，并展示
for i in range(min(3, len(booster.get_dump()))):  # 只展示前3棵树
    print(f"Tree {i+1} Structure:")
    print(booster.get_dump()[i])

# 如果想要可视化树结构（例如第一个树），可以使用plot_tree方法
# plt.figure(figsize=(20, 10))
# xgb.plot_tree(model, num_trees=0)  # 绘制第一棵树
# plt.show()
