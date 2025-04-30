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

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

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
shap_value_for_instance = shap_values[idx]
print(f"SHAP values for instance {idx}: {shap_value_for_instance.values}")

# 可视化该实例的SHAP值
shap.initjs()  # 初始化JS可视化工具
shap.force_plot(shap_values[idx].base_values, shap_values[idx].values, X_test[idx], feature_names=data.columns[:-1])

# 如果需要查看所有样本的SHAP值，可使用summary_plot
shap.summary_plot(shap_values.values, X_test, feature_names=data.columns[:-1])

# 保存SHAP图像
shap.force_plot(shap_values[idx].base_values, shap_values[idx].values, X_test[idx], feature_names=data.columns[:-1])
plt.savefig('shap_explanation.png')  # 保存为图片
