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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# 设定参数
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42

# subsection 选用的大规模数据集
file_path = "../../../datasets/multi_class/adult.csv"
data = pd.read_csv(file_path)

# subsection 选用的大规模数据集
# file_path = "../../../datasets/multi_class/flights.csv"
# data = pd.read_csv(file_path).dropna(subset=['WEATHER_DELAY'])

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
outlier_indices = np.where(y_pred == 1)[0][:10]  # 前10个预测为离群值的样本

# 使用XGBoost的plot_importance来展示特征的重要性
"""plot_importance将展示各个特征对模型预测的重要性"""
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
