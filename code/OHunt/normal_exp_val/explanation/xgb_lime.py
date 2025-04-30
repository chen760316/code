"""
 LIME 解释：
    使用 LimeTabularExplainer 创建 LIME 解释器对象，来解释 XGBoost 模型对个别样本的预测。
    LIME 提供了模型的局部解释，通过 explain_instance 函数，我们可以解释模型如何做出某个预测。
    num_features=10 表示我们选择显示影响预测结果的前 10 个特征。
    结果可以以表格或图像的形式进行可视化。
LIME 解释结果：
    在解释过程中，exp.show_in_notebook 会显示该实例的局部解释，解释模型的预测结果是如何由输入特征决定的。
    通过 exp.as_pyplot_figure()，我们将解释结果绘制成图像并展示。
解释模型输出：
    LIME解释 会显示该样本的预测结果与每个特征的贡献。这可以帮助你了解模型是如何对特定的离群样本做出分类决策的，哪些特征对最终的预测结果贡献最大。
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

# 使用LIME解释模型
# 初始化LIME解释器
# 获取特征名
feature_names = data.columns[:-1]  # 获取特征列名，不包括最后一列标签
explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode="classification", discretize_continuous=True, feature_names=feature_names)

# 选择一个预测为离群值的样本（通常我们选择预测为1的离群样本）
idx = np.where(y_pred == 1)[0][0]  # 选择第一个预测为离群样本的索引
print(f"Explaining prediction for instance {idx}")

# 对该样本进行LIME解释
exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=10)

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