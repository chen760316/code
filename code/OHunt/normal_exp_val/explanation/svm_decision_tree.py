"""
训练决策树：我将 LimeTabularExplainer 替换为 DecisionTreeClassifier，并训练该决策树来解释 SVM 模型对样本的预测。
决策树可视化：使用 plot_tree 来可视化训练后的决策树。决策树的可视化图可以帮助理解它如何做出分类决策。
决策路径：可以进一步探索决策树如何对特定的样本进行分类决策（这通常会通过访问 decision_path 方法实现，尤其是在解释决策树对某个样本的预测时）。
这里的决策树并不一定完全符合 SVM 模型的决策，但它给出了一个局部的、简化的解释。对于一个具体的样本，决策树通过其结构展示了如何在输入特征的基础上做出决策。

TreeExplainer 是 SHAP 中专门用于树模型（如 XGBoost、LightGBM）的解释器。
shap.summary_plot 会生成一个特征重要性的汇总图，展示各特征对模型预测的贡献。
shap.force_plot 显示单个实例的具体预测贡献，帮助我们了解每个特征是如何影响模型决策的。
"""

import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib
matplotlib.use('TkAgg')  # 或者尝试 'Agg'，它是非交互式的
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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

# 初始化SVM模型
model = SVC(kernel='rbf', random_state=random_state, probability=True)

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

# 训练一个决策树解释器来解释XGBoost的预测
tree_explainer = DecisionTreeClassifier(random_state=random_state)
tree_explainer.fit(X_train, y_train)

# 使用决策树模型预测
tree_pred = tree_explainer.predict(X_test)

# 获取该样本的决策树预测结果
decision_tree_prediction = tree_pred[idx]
print(f"Decision Tree prediction for instance {idx}: {decision_tree_prediction}")

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(tree_explainer, filled=True, feature_names=data.columns[:-1], class_names=['Class 0', 'Class 1'])
plt.title("Decision Tree Visualization")
plt.show()

# 如果需要查看决策树在特定样本上的决策路径，可以直接访问其决策过程
# 通过访问决策树的路径进行解释
node_indicator = tree_explainer.decision_path(X_test[idx].reshape(1, -1))
print(f"Decision tree path for sample {idx}: {node_indicator}")
