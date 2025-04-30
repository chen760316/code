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
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# subsection 选用的大规模数据集
# file_path = "../../../datasets/multi_class/adult.csv"
# data = pd.read_csv(file_path)

# subsection 选用的大规模数据集
file_path = "../../../datasets/multi_class/flights.csv"
data = pd.read_csv(file_path).dropna(axis=1, how='all')

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

data = data.fillna(data.mean())  # 例如，用均值填充

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

# 定义一个超参数网格
param_grid = {
    'C': [0.1, 1.0, 10],  # 正则化参数
    'kernel': ['linear'],  # 核函数类型
    'class_weight': ['balanced', None],  # 类别权重
    'gamma': ['scale', 'auto']  # 仅对 rbf 核有用，linear 核忽略
}

# 初始化 SVM 模型（带概率估计）
model = svm.SVC(probability=True, random_state=42)

# 初始化 GridSearchCV 进行超参数调优，使用验证集进行交叉验证
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=2,  # 3折交叉验证（你原本写的是 3）
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)

# 在验证集上进行网格搜索
grid_search.fit(X_val, y_val)

# 输出最佳参数和最佳交叉验证得分
print("最佳超参数组合：", grid_search.best_params_)
print("最佳交叉验证准确度：", grid_search.best_score_)

# 使用最佳超参数训练模型
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

# 1. 支持向量（Support Vectors）
support_vectors_idx = model.support_  # 获取支持向量的索引
print(f"Number of Support Vectors: {len(support_vectors_idx)}")
print("支持向量的索引：", support_vectors_idx)

# 显示支持向量的样本
X_support_vectors = X_train.iloc[support_vectors_idx]
y_support_vectors = y_train.iloc[support_vectors_idx]
print(f"Support Vector Samples: {X_support_vectors}")

# 检查模型是否具有 coef_ 属性
if hasattr(model, 'coef_'):
    feature_weights = model.coef_[0]  # 获取特征的权重（系数）
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

else:
    print("The model does not have coef_ attribute. Please use a linear kernel (kernel='linear').")

