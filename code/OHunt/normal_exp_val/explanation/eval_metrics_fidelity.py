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

"""
计算忠诚度：
在忠实度计算过程中，我们关注的是原始模型预测的概率与基于LIME解释的特征构建的输入预测的概率之间的差异。
因此，通过 model.predict_proba 获取预测的概率值，并计算它们之间的均方误差（MSE）。

解释：
忠实度的计算：忠实度实际上是度量解释与模型预测之间的一致性。在这里，我们通过LIME解释器选出的特征来重建模型的预测行为，而不直接使用真实标签来计算忠实度。
真实标签 y_instance 可以用来衡量模型预测的准确性，但忠实度更多的是衡量模型在被LIME解释时的预测稳定性和一致性。

Fidelity得分为0的常见原因可能是：模型的真实预测与修改后的输入非常相似，导致均方误差接近0。特别是当模型对特征变化不敏感时，这种情况会更明显。
"""

X_instance = X_test[idx]

# 计算忠实度
fidelity_score = fidelity(model, explainer, X_instance, num_features=10)
print(f"Fidelity Score for instance {idx}: {fidelity_score:.4f}")