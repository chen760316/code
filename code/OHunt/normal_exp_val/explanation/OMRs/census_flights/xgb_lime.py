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

X = data.values[:, :-1]
y = data.values[:, -1]
# X = StandardScaler().fit_transform(X)

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

# 使用LIME解释模型
# 初始化LIME解释器
# 获取特征名
feature_names = data.columns[:-1]  # 获取特征列名，不包括最后一列标签
explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode="classification", discretize_continuous=False, feature_names=feature_names)

# 选择一个预测为离群值的样本（通常我们选择预测为1的离群样本）
idx = np.where(y_pred == 1)[0][0]  # 选择第一个预测为离群样本的索引
print(f"Explaining prediction for instance {idx}")
outlier_indices = np.where(y_pred == 1)[0][:10]  # 前10个预测为离群值的样本

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

"""
Weight（权重）：
    Weight 表示每个特征在模型局部决策中所占的“重要性”或“贡献度”。
    具体来说，Weight 是该特征对目标预测值的贡献大小。对于一个样本，特征的权重决定了该特征在模型最终决策中的重要性。
        如果一个特征的权重大（正或负），说明它在影响该样本的预测时起到了较大的作用。
        如果一个特征的权重接近于0，则说明该特征对该样本的预测几乎没有影响。
    权重值的符号（正负）也很重要：
        正权重：该特征的值越大，预测结果越倾向于某一类别（例如类别1）。
因为LIME是一个局部解释方法，所以这些 Weight 仅仅是对目标样本的预测进行解释的权重，而不是全局性（即对所有样本）的特征重要性。
"""

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


from sklearn.metrics import mean_squared_error
import re

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
sparsity_scores = 0
for idx in outlier_indices:
    X_instance = X_test[idx]

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



