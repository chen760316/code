"""
𝑅(𝑡) ∧ outlier(𝐷, 𝑅, 𝑡.𝐴, 𝜃) ∧ loss(M, D, 𝑡) > 𝜆 ∧ M𝑐 (𝑅, 𝐴,M) → ugly(𝑡)
Rovas对ugly outliers的检测能力
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

# subsection 原始真实数据集（对应实验测试1.1）

# file_path = "../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../datasets/real_outlier/annthyroid.csv"
# file_path = "../datasets/real_outlier/optdigits.csv"
# file_path = "../datasets/real_outlier/PageBlocks.csv"
# file_path = "../datasets/real_outlier/pendigits.csv"
# file_path = "../datasets/real_outlier/satellite.csv"
# file_path = "../datasets/real_outlier/shuttle.csv"
file_path = "../datasets/real_outlier/yeast.csv"

# subsection 含有不同异常比例的真实数据集（对应实验测试1.2）

# choice Annthyroid数据集
# file_path = "../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_02_v01.csv"
# file_path = "../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_05_v01.csv"
# file_path = "../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_07.csv"

# choice Cardiotocography数据集
# file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_02_v01.csv"
# file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_05_v01.csv"
# file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_10_v01.csv"
# file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_20_v01.csv"
# file_path = "../datasets/real_outlier_varying_ratios/Cardiotocography/Cardiotocography_22.csv"

# choice PageBlocks数据集
# file_path = "../datasets/real_outlier_varying_ratios/PageBlocks/PageBlocks_02_v01.csv"
# file_path = "../datasets/real_outlier_varying_ratios/PageBlocks/PageBlocks_05_v01.csv"

# choice Wilt数据集
# file_path = "../datasets/real_outlier_varying_ratios/Wilt/Wilt_02_v01.csv"
# file_path = "../datasets/real_outlier_varying_ratios/Wilt/Wilt_05.csv"

# subsection 含有不同异常类型和异常比例的合成数据集（从真实数据中加入不同异常类型合成）（对应实验测试1.2）

# choice Annthyroid数据集+cluster噪声+不同噪声比例(效果稳定)
# file_path = "../datasets/synthetic_outlier/annthyroid_cluster_0.1.csv"
# file_path = "../datasets/synthetic_outlier/annthyroid_cluster_0.2.csv"
# file_path = "../datasets/synthetic_outlier/annthyroid_cluster_0.3.csv"

# choice Cardiotocography数据集+local噪声+不同噪声比例(好用)
# file_path = "../datasets/synthetic_outlier/Cardiotocography_local_0.1.csv"
# file_path = "../datasets/synthetic_outlier/Cardiotocography_local_0.2.csv"
# file_path = "../datasets/synthetic_outlier/Cardiotocography_local_0.3.csv"

# choice PageBlocks数据集+global噪声+不同噪声比例(效果稳定)
# file_path = "../datasets/synthetic_outlier/PageBlocks_global_0.1.csv"
# file_path = "../datasets/synthetic_outlier/PageBlocks_global_0.2.csv"
# file_path = "../datasets/synthetic_outlier/PageBlocks_global_0.3.csv"

# choice satellite数据集+local噪声+不同噪声比例(好用)
# file_path = "../datasets/synthetic_outlier/satellite_0.1.csv"
# file_path = "../datasets/synthetic_outlier/satellite_0.2.csv"
# file_path = "../datasets/synthetic_outlier/satellite_0.3.csv"

# choice annthyroid数据集+local噪声+不同噪声比例(好用)
# file_path = "../datasets/synthetic_outlier/annthyroid_0.1.csv"
# file_path = "../datasets/synthetic_outlier/annthyroid_0.2.csv"
# file_path = "../datasets/synthetic_outlier/annthyroid_0.3.csv"

# choice waveform数据集+dependency噪声+不同噪声比例
# file_path = "../datasets/synthetic_outlier/waveform_dependency_0.1.csv"
# file_path = "../datasets/synthetic_outlier/waveform_dependency_0.2.csv"
# file_path = "../datasets/synthetic_outlier/waveform_dependency_0.3.csv"

data = pd.read_csv(file_path)

# 如果数据量超过20000行，就随机采样到20000行
if len(data) > 20000:
    data = data.sample(n=20000, random_state=42)

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

X = data.values[:, :-1]
y = data.values[:, -1]

# 统计不同值及其数量
unique_values, counts = np.unique(y, return_counts=True)

# 输出结果
for value, count in zip(unique_values, counts):
    print(f"标签: {value}, 数量: {count}")

# 找到最小标签的数量
min_count = counts.min()
total_count = counts.sum()

# 计算比例
proportion = min_count / total_count
print(f"较少标签占据的比例: {proportion:.4f}")
min_count_index = np.argmin(counts)  # 找到最小数量的索引
min_label = unique_values[min_count_index]  # 对应的标签值

all_columns = data.columns.values.tolist()
feature_names = all_columns[:-1]
class_name = all_columns[-1]

# 找到分类特征的列名
categorical_columns = data.select_dtypes(exclude=['float']).columns[:-1]
# 获取分类特征对应的索引
categorical_features = [data.columns.get_loc(col) for col in categorical_columns]

# section 数据特征缩放以及添加噪声
# section 划分训练集，验证集，测试集，划分比例葳7;2:1

# 数据标准化（对所有特征进行标准化）
X = StandardScaler().fit_transform(X)

# 记录原始索引
original_indices = np.arange(len(X))
# 按照 70% 训练集，20% 验证集，10% 测试集的比例随机划分数据集
X_train, X_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)
# 从临时数据集（30%）中划分出 10% 测试集和 20% 验证集
X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(X_temp, y_temp, temp_indices, test_size=0.33, random_state=1)
print(f"Training Set: {X_train.shape}, Validation Set: {X_val.shape}, Test Set: {X_test.shape}")

# 加入随机噪声的比例
noise_level = 0.2
# 计算噪声数量
n_samples = X.shape[0]
n_noise = int(noise_level * n_samples)
# 随机选择要添加噪声的样本
noise_indices = np.random.choice(n_samples, n_noise, replace=False)
# 添加高斯噪声到特征
X_copy = np.copy(X)
X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
# 从加噪数据中生成加噪训练数据和加噪测试数据
X_train_copy = X_copy[train_indices]
X_val_copy = X_copy[val_indices]
X_test_copy = X_copy[test_indices]
# 创建 DataFrame 存储加噪数据集 D'（用于后续分析）
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
all_columns = list(data.columns)  # 假设您已经有了数据的原始列名
data_copy = pd.DataFrame(combined_array, columns=all_columns)
# 训练集中添加了高斯噪声的样本在原始数据集 D 中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集 D 中的索引
test_noise = np.intersect1d(test_indices, noise_indices)

# section 找到有影响力的特征 M𝑐 (𝑅, 𝐴, M)
# section 没暴露测试集
# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
import re

i = len(feature_names)
np.random.seed(1)
categorical_names = {}
# 定义一个超参数网格
param_grid = {
    'n_estimators': [50, 100, 150],  # 树的数量
    'max_depth': [5, 10, 15, None],   # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
    'min_samples_leaf': [1, 2, 4],    # 叶节点的最小样本数
    'max_features': ['auto', 'sqrt', 'log2'],  # 最大特征数
    'class_weight': ['balanced', None]  # 类别权重
}

# 初始化随机森林模型
rf_model = RandomForestClassifier(random_state=42)

# 初始化GridSearchCV进行超参数调优，使用验证集进行交叉验证
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=3,  # 5折交叉验证
                           scoring='accuracy',  # 使用准确度作为评分指标
                           n_jobs=-1,  # 使用所有CPU核进行并行计算
                           verbose=2)  # 打印详细的进度信息

# 在验证集上进行网格搜索
grid_search.fit(X_val, y_val)

# 输出最佳参数和最佳交叉验证得分
print("最佳超参数组合：", grid_search.best_params_)
print("最佳交叉验证准确度：", grid_search.best_score_)

# 使用最佳超参数训练模型
rf_model_noise = grid_search.best_estimator_

rf_model_noise.fit(X_train_copy, y_train)

for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data_copy.iloc[:, feature])
    data_copy.iloc[:, feature] = le.transform(data_copy.iloc[:, feature])
    categorical_names[feature] = le.classes_

explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)

predict_fn = lambda x: rf_model_noise.predict_proba(x)
exp = explainer.explain_instance(X_train[i], predict_fn, num_features=len(feature_names)//2)
# 获取最具影响力的特征及其权重
top_features = exp.as_list()
top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
top_k_indices = [feature_names.index(name) for name in top_feature_names]
print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

# section 找到loss(M, D, 𝑡) > 𝜆的元组
# section 没暴露测试集

# choice 使用交叉熵损失函数
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

y_mask = y.copy()
y_mask[test_indices] = rf_model_noise.predict(X_test_copy)
# 对y_ground进行独热编码
encoder = OneHotEncoder(sparse_output=False)
y_true = encoder.fit_transform(y.reshape(-1, 1))

# 获取每棵树的预测概率 (n_samples, n_classes)
probabilities = rf_model_noise.predict_proba(X_copy)

# 假设我们需要对每个样本在所有类别上的分数应用 Softmax
# 对于每个样本，随机森林会输出多个类别的概率，reshape 后应用 Softmax
y_pred = softmax(probabilities, axis=1)

# 计算每个样本的损失
loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
# 使用模型进行预测
y_pred_copy = rf_model_noise.predict(X_copy)
# 计算分类错误率
mis_classification_rate = 1 - accuracy_score(y, y_pred_copy)
bad_num = int(mis_classification_rate * len(X_copy))
# 计算测试集平均多分类交叉熵损失
average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
# 获取差距最小的num个样本的索引
valid_samples_indices = np.where(loss_per_sample > average_loss)[0]
# 计算这些样本与average_loss的差距
loss_difference = np.abs(loss_per_sample[valid_samples_indices] - average_loss)
# 获取与average_loss差距最小的num个样本的索引
bad_samples = valid_samples_indices[np.argsort(loss_difference)[-bad_num:]]
ugly_outlier_candidates = bad_samples

# 计算测试集平均多分类交叉熵损失
# bad_samples = np.where(loss_per_sample > average_loss)[0]
# good_samples = np.where(loss_per_sample < average_loss)[0]
# ugly_outlier_candidates = bad_samples


# choice 使用二元hinge损失函数
# y_mask = y.copy()
# y_mask[test_indices] = rf_model.predict(X_test_copy)
# predictions = rf_model.decision_function(X_copy)
# y_pred = np.where(predictions < 0, 0, 1)
# bad_samples = np.where(y_pred != y_mask)[0]
# ugly_outlier_candidates = bad_samples

# section 谓词outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )的实现，找到所有有影响力的特征下的异常元组
# section 没暴露测试集

outlier_feature_indices = {}
threshold = 0.15
for column_indice in top_k_indices:
    data_normalization = data_copy.copy()
    # 初始化一个 MinMaxScaler
    scaler = MinMaxScaler()
    # 对 DataFrame 进行归一化，注意这是按列操作的
    data_normalization = pd.DataFrame(scaler.fit_transform(data_normalization), columns=data_normalization.columns)
    select_feature = feature_names[column_indice]
    # select_column_data = data_normalization[select_feature].loc[train_indices].values
    select_column_data = data_normalization[select_feature].values
    max_value = np.max(select_column_data)
    min_value = np.min(select_column_data)
    sorted_indices = np.argsort(select_column_data)
    sorted_data = select_column_data[sorted_indices]
    # 找到A属性下的所有异常值
    outliers = []
    outliers_index = []
    # 检查列表首尾元素
    if len(sorted_data) > 1:
        if (sorted_data[1] - sorted_data[0] >= threshold):
            outliers.append(sorted_data[0])
            outliers_index.append(sorted_indices[0])
        if (sorted_data[-1] - sorted_data[-2] >= threshold):
            outliers.append(sorted_data[-1])
            outliers_index.append(sorted_indices[-1])
    # 检查中间元素
    for i in range(1, len(sorted_data) - 1):
        current_value = sorted_data[i]
        left_value = sorted_data[i - 1]
        right_value = sorted_data[i + 1]
        if (current_value - left_value >= threshold) and (right_value - current_value >= threshold):
            outliers.append(current_value)
            outliers_index.append(sorted_indices[i])
    outliers_index_numpy = np.array(outliers_index)
    intersection = np.array(outliers_index)
    # intersection = np.intersect1d(np.array(outliers_index), ugly_outlier_candidates)
    # print("有影响力的特征A下同时满足outlier(𝐷, 𝑅, 𝑡 .𝐴, 𝜃 )和loss(M, D, 𝑡) > 𝜆的所有异常值索引为：", intersection)
    outlier_feature_indices[column_indice] = intersection
# print(outlier_feature_indices)

# section 确定数据中的ugly outliers
# section 没暴露测试集

outlier_tuple_set = set()
for value in outlier_feature_indices.values():
    outlier_tuple_set.update(value)
outlier_tuple_set.update(bad_samples)
X_copy_repair_indices = list(outlier_tuple_set)
X_copy_repair = X_copy[X_copy_repair_indices]
y_repair = y[X_copy_repair_indices]

# 生成保留的行索引
rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

# 使用保留的行索引选择D'中的正常数据
# 无需修复的特征和标签值
X_copy_inners = X_copy[rows_to_keep]
y_inners = y[rows_to_keep]

# section 训练下游任务的random_forest模型
# section 没暴露测试集

# subsection 原始数据集上训练的random_forest模型在训练集和测试集中分错的样本比例
# subsection 没暴露测试集

print("*" * 100)
rf_model = grid_search.best_estimator_
rf_model.fit(X_train, y_train)
train_label_pred = rf_model.predict(X_train)

# 训练样本中被random_forest模型错误分类的样本
wrong_classified_train_indices = np.where(y_train != rf_model.predict(X_train))[0]
print("训练样本中被random_forest模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

# 测试样本中被random_forest模型错误分类的样本
wrong_classified_test_indices = np.where(y_test != rf_model.predict(X_test))[0]
print("测试样本中被random_forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

# 整体数据集D中被random_forest模型错误分类的样本
print("完整数据集D中被random_forest模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

# subsection 加噪数据集上训练的random_forest模型在训练集和测试集中分错的样本比例
# subsection 没暴露测试集

print("*" * 100)
train_label_pred_noise = rf_model_noise.predict(X_train_copy)

# 加噪训练样本中被random_forest模型错误分类的样本
wrong_classified_train_indices_noise = np.where(y_train != rf_model_noise.predict(X_train_copy))[0]
print("加噪训练样本中被random_forest模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise)/len(y_train))

# 加噪测试样本中被random_forest模型错误分类的样本
wrong_classified_test_indices_noise = np.where(y_test != rf_model_noise.predict(X_test_copy))[0]
print("加噪测试样本中被random_forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise)/len(y_test))

# 整体加噪数据集D中被random_forest模型错误分类的样本
print("完整数据集D中被random_forest模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

# section 全部加噪数据中被random forest分类器误分类的数量
# section 没暴露测试集

label_pred = rf_model_noise.predict(X_copy)
wrong_classify_indices = []
for i in range(len(X_copy)):
    if y[i] != label_pred[i]:
        wrong_classify_indices.append(i)
print("被误分类的样本数量：", len(wrong_classify_indices))

# section 检测ugly outliers的召回率
# section 没暴露测试集

# ugly_found_by_detector = list(set(X_copy_repair_indices) & set(wrong_classify_indices))
ugly_found_by_detector = list(set(X_copy_repair_indices) & set(wrong_classify_indices))
print("召回的ugly outliers的数量：", len(ugly_found_by_detector))
print("ugly outliers的召回率为：", len(ugly_found_by_detector)/len(wrong_classify_indices))

# section 重新计算recall/precision/F1分数
# 计算 TP, FN, FP
TP = len(set(X_copy_repair_indices) & set(wrong_classify_indices))  # 交集元素数量
FN = len(set(wrong_classify_indices) - set(X_copy_repair_indices))  # s2中有但s1中没有的元素数量
FP = len(set(X_copy_repair_indices) - set(wrong_classify_indices))  # s1中有但s2中没有的元素数量

# 计算召回率 (Recall)
Recall = TP / (TP + FN) if (TP + FN) != 0 else 0

# 计算精确度 (Precision)
Precision = TP / (TP + FP) if (TP + FP) != 0 else 0

# 计算 F1 分数
F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0

# 打印结果
print("*"*100)
print("候选的ugly outliers列表长度为：", len(X_copy_repair_indices))
print("真实的ugly outliers列表长度为：", len(wrong_classify_indices))
print(f"Recall: {Recall:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"F1 Score: {F1:.4f}")