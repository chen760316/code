"""
发现检测ugly outliers 的RoDs规则
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

from deepod.models.tabular import GOAD
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from deepod.models.tabular import DeepSVDD
from deepod.models.tabular import RCA
from deepod.models import REPEN, SLAD, ICL, NeuTraL
import re
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
import optuna
import os
import time
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 数据预处理
# file_path = "../large_datasets/sampled_flights.csv"
# data = pd.read_csv(file_path)
#
# if len(data) > 30000:
#     data = data.sample(n=30000, random_state=42)
#
# # 删除包含任何 NaN 的列
# data = data.dropna(axis=1)
file_path = "../../large_datasets/samples/flights_10000.csv"
data = pd.read_csv(file_path)

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

# 找到分类特征的列名
categorical_columns = data.select_dtypes(exclude=['float']).columns[:-1]
# 获取分类特征对应的索引
categorical_features = [data.columns.get_loc(col) for col in categorical_columns]

# # 统计不同值及其数量
# unique_values, counts = np.unique(y, return_counts=True)
#
# # 输出结果
# for value, count in zip(unique_values, counts):
#     print(f"标签: {value}, 数量: {count}")
#
# # 找到最小标签的数量
# min_count = counts.min()
# total_count = counts.sum()
#
# # 计算比例
# proportion = min_count / total_count
# print(f"较少标签占据的比例: {proportion:.4f}")
# min_count_index = np.argmin(counts)  # 找到最小数量的索引
# min_label = unique_values[min_count_index]  # 对应的标签值

# section 数据特征缩放

# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = \
    train_test_split(X, y, original_indices, test_size=0.3, random_state=42)

# 第二次分割：从训练集中再抽 20% 做验证集
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
# 从含噪数据中生成训练数据和测试数据
X_train_copy = X_copy[train_indices]
X_test_copy = X_copy[test_indices]
feature_names = data.columns.values.tolist()
combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
# 添加噪声后的数据集D'对应的Dataframe
data_copy = pd.DataFrame(combined_array, columns=feature_names)
# 训练集中添加了高斯噪声的样本在原始数据集D中的索引
train_noise = np.intersect1d(train_indices, noise_indices)
# 测试集中添加了高斯噪声的样本在原始数据集D中的索引
test_noise = np.intersect1d(test_indices, noise_indices)

print(data_copy.index)

# SECTION SVM模型的实现

# # subsection 原始数据集上训练的SVM模型在训练集和测试集中分错的样本比例

# subsection 加噪数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
# 定义超参数网格
param_grid = {
    'C': [0.1, 1.0, 10],  # 正则化参数
    'kernel': ['linear', 'rbf'],  # 核函数类型
    'class_weight': ['balanced', None],  # 类别权重
    'gamma': ['scale', 'auto']  # 对于 rbf 核有用
}

# 初始化 SVM 模型（带概率估计）
svm_model_noise = svm.SVC(probability=True, random_state=42)

# 初始化 GridSearchCV 进行超参数调优
grid_search = GridSearchCV(estimator=svm_model_noise, param_grid=param_grid,
                           cv=3,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)

# 拟合模型（在验证集上进行网格搜索）
grid_search.fit(X_val, y_val)

# 输出最佳参数和交叉验证得分
print("最佳超参数组合：", grid_search.best_params_)
print("最佳交叉验证准确度：", grid_search.best_score_)

# 使用最佳模型
svm_model_noise = grid_search.best_estimator_
svm_model_noise.fit(X_train_copy, y_train)
train_label_pred_noise = svm_model_noise.predict(X_train_copy)
test_label_pred_noise = svm_model_noise.predict(X_test_copy)



# SECTION 检测有影响力的特征MDetO(𝑡,𝐴,D)的实现
# choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
def MDetI(clf, data_copy, X_train_copy):
    feature_names = data_copy.columns.values.tolist()

    # 找到分类特征的列名
    categorical_columns = data_copy.select_dtypes(exclude=['float']).columns[:-1]
    # 获取分类特征对应的索引
    categorical_features = [data_copy.columns.get_loc(col) for col in categorical_columns]

    i = len(feature_names)
    np.random.seed(1)
    categorical_names = {}
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data_copy.iloc[:, feature])
        data_copy.iloc[:, feature] = le.transform(data_copy.iloc[:, feature])
        categorical_names[feature] = le.classes_
    explainer = LimeTabularExplainer(X_train_copy, feature_names=feature_names, class_names=feature_names,
                                     categorical_features=categorical_features,
                                     categorical_names=categorical_names, kernel_width=3)
    # predict_proba 方法用于分类任务，predict 方法用于回归任务
    predict_fn = lambda x: clf.predict_proba(x)
    exp = explainer.explain_instance(X_train_copy[i], predict_fn, num_features=len(feature_names) // 2)
    # 获取最具影响力的特征及其权重
    top_features = exp.as_list()
    top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
    top_k_indices = [feature_names.index(name) for name in top_feature_names]
    return top_k_indices

# SECTION MDetO(𝑡,𝐴,D) 针对元组异常的无监督异常检测器GOAD的实现
def MDetO(outlier_detector, X_train_copy, X_test_copy, X_copy):
    epochs = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_trans = 64
    random_state = 42

    if outlier_detector == "GOAD":
        out_clf_noise = GOAD(epochs=epochs, device=device, n_trans=n_trans)
        out_clf_noise.fit(X_train_copy, y=None)
    elif outlier_detector == "RCA":
        out_clf_noise = RCA(epochs=epochs, device=device, act='LeakyReLU')
        out_clf_noise.fit(X_train_copy, y=None)
    elif outlier_detector == "DeepSVDD":
        out_clf_noise = DeepSVDD(epochs=epochs, device=device, random_state=random_state)
        out_clf_noise.fit(X_train_copy, y=None)
    elif outlier_detector == "RePEN":
        out_clf_noise = REPEN(epochs=epochs, device=device)
        out_clf_noise.fit(X_train_copy, y=None)
    elif outlier_detector == "ICL":
        out_clf_noise = ICL(epochs=epochs, device=device, n_ensemble='auto')
        out_clf_noise.fit(X_train_copy, y=None)
    elif outlier_detector == "NeuTraL":
        out_clf_noise = NeuTraL(epochs=epochs, device=device)
        out_clf_noise.fit(X_train_copy, y=None)
    else:
        out_clf_noise = SLAD(epochs=epochs, device=device)
        out_clf_noise.fit(X_train_copy, y=None)

    # 从加噪训练集中检测出异常值索引
    train_pred_labels_noise, train_confidence_noise = out_clf_noise.predict(X_train_copy, return_confidence=True)
    print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
    train_outliers_index_noise = []
    print("加噪训练集样本数：", len(X_train_copy))
    for i in range(len(X_train_copy)):
        if train_pred_labels_noise[i] == 1:
            train_outliers_index_noise.append(i)
    print("加噪训练集中异常值索引：", train_outliers_index_noise)

    # 从加噪测试集中检测出异常值索引
    test_pred_labels_noise, test_confidence_noise = out_clf_noise.predict(X_test_copy, return_confidence=True)
    print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
    test_outliers_index_noise = []
    print("加噪测试集样本数：", len(X_test_copy))
    for i in range(len(X_test_copy)):
        if test_pred_labels_noise[i] == 1:
            test_outliers_index_noise.append(i)
    # 训练样本中的异常值索引
    print("加噪测试集中异常值索引：", test_outliers_index_noise)

    # 从整体的加噪数据集中检测出异常值索引
    pred_labels_noise, onfidence_noise = out_clf_noise.predict(X_copy, return_confidence=True)
    print("加噪数据集中异常值判定阈值为：", out_clf_noise.threshold_)
    outliers_index_noise = []
    print("加噪数据集样本数：", len(X_copy))
    for i in range(len(X_copy)):
        if pred_labels_noise[i] == 1:
            outliers_index_noise.append(i)
    # 训练样本中的异常值索引
    print("加噪数据集中异常值索引：", outliers_index_noise)

    return train_outliers_index_noise, test_outliers_index_noise, outliers_index_noise

# section outlier(𝐷,𝑅,𝑡,𝐴,𝜃)的实现
def outlier(data_copy, theta_threshold, top_k_indices, ugly_outlier_candidates):
    outlier_feature_indices = {}
    threshold = theta_threshold
    feature_names = data_copy.columns.values.tolist()
    for column_indice in top_k_indices:
        select_feature = feature_names[column_indice]
        select_column_data = data_copy[select_feature].values
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
        intersection = np.intersect1d(np.array(outliers_index), ugly_outlier_candidates)
        outlier_feature_indices[column_indice] = intersection
        outlier_tuple_set = set()
        for value in outlier_feature_indices.values():
            outlier_tuple_set.update(value)
        X_copy_repair_indices = list(outlier_tuple_set)
        return X_copy_repair_indices

# section loss(M,D,𝑡,𝐴)的实现
def loss(clf, X_copy, X_test_copy, y_train, loss_choice):

    # choice 使用sklearn库中的hinge损失函数
    if loss_choice == "hinge":
        decision_values = clf.decision_function(X_copy)
        # 计算每个样本的hinge损失
        num_samples = X_copy.shape[0]
        num_classes = clf.classes_.shape[0]
        hinge_losses = np.zeros((num_samples, num_classes))
        hinge_loss = np.zeros(num_samples)
        for i in range(num_samples):
            correct_class = int(y[i])
            for j in range(num_classes):
                if j != correct_class:
                    loss_j = max(0, 1 - decision_values[i, correct_class] + decision_values[i, j])
                    hinge_losses[i, j] = loss_j
            hinge_loss[i] = np.max(hinge_losses[i])
        # 计算 hinge_loss 的均值
        mean_hinge_loss = np.mean(hinge_loss)
        print("所有样本的hinge loss的均值：", mean_hinge_loss)
        # 在所有加噪数据D中损失函数高于阈值的样本索引
        ugly_outlier_candidates = np.where(hinge_loss > 1)[0]

    # choice 使用交叉熵损失函数(适用于二分类数据集下的通用分类器，判定bad不够准确)
    elif loss_choice == "cross_entropy":
        # 获取决策值
        decision_values = clf.decision_function(X_copy)
        # 将决策值转换为适用于 Softmax 的二维数组
        decision_values_reshaped = decision_values.reshape(-1, 1)  # 变成 (n_samples, 1)
        # 应用 Softmax 函数（可以手动实现或使用 scipy）
        y_pred = softmax(np.hstack((decision_values_reshaped, -decision_values_reshaped)), axis=1)
        # 创建 OneHotEncoder 实例
        encoder = OneHotEncoder(sparse_output=False)
        # 预测y_test的值，并与y_train组合成为y_ground
        y_test_pred = clf.predict(X_test_copy)
        y_ground = np.hstack((y_train, y_test_pred))
        # 对y_ground进行独热编码
        y_true = encoder.fit_transform(y_ground.reshape(-1, 1))
        # 计算每个样本的损失
        loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
        # 计算测试集平均多分类交叉熵损失
        average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
        bad_samples = np.where(loss_per_sample > average_loss)[0]
        good_samples = np.where(loss_per_sample <= average_loss)[0]
        # 在所有加噪数据D中损失函数高于阈值的样本索引
        # ugly_outlier_candidates = np.where(bad_samples > 1)[0]
        ugly_outlier_candidates = bad_samples

    # choice 使用多分类交叉熵损失函数
    elif loss_choice == "multi_cross_entropy":
        # 获取训练集和测试集的决策值 (logits)
        decision_values_train = clf.decision_function(X_copy)
        decision_values_test = clf.decision_function(X_test_copy)

        # 对决策值应用 Softmax（scipy 中实现的 Softmax）
        y_pred_train = softmax(decision_values_train, axis=1)
        y_pred_test = softmax(decision_values_test, axis=1)

        # 创建 OneHotEncoder 实例
        encoder = OneHotEncoder(sparse_output=False)

        # 对训练集标签进行独热编码
        y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))

        # 预测测试集标签
        y_test_pred = clf.predict(X_test_copy)
        y_test_onehot = encoder.transform(y_test_pred.reshape(-1, 1))

        # 计算训练集的交叉熵损失 per sample
        loss_per_sample_train = -np.sum(y_train_onehot * np.log(y_pred_train + 1e-12), axis=1)

        # 计算测试集的交叉熵损失 per sample
        loss_per_sample_test = -np.sum(y_test_onehot * np.log(y_pred_test + 1e-12), axis=1)

        # 合并训练集和测试集的损失
        loss_per_sample = np.concatenate([loss_per_sample_train, loss_per_sample_test])

        # 计算平均交叉熵损失
        average_loss = np.mean(loss_per_sample)

        # 找出损失较大的样本，作为“坏样本”
        bad_samples = np.where(loss_per_sample > average_loss)[0]
        good_samples = np.where(loss_per_sample <= average_loss)[0]

        # 找到所有损失较大的样本的索引，可能是“离群样本”
        ugly_outlier_candidates = bad_samples

    # choice 直接判断
    else:
        y_pred = clf.predict(X_copy)
        ugly_outlier_candidates = np.where(y_pred != y)[0]
        # 提取对应索引的标签
        selected_labels = y[ugly_outlier_candidates]
        print("ugly_outlier_candidates的数量：", len(ugly_outlier_candidates))
        print("ugly_outlier_candidates中标签为1的样本数量：", np.sum(selected_labels == 1))

    return ugly_outlier_candidates

# section imbalanced(𝐷,𝑅,𝑡,𝐴,𝛿)的实现
def calculate_made(data_copy):
    median = np.median(data_copy)  # 计算中位数
    abs_deviation = np.abs(data_copy - median)  # 计算每个数据点与中位数的绝对误差
    mad = np.median(abs_deviation)  # 计算绝对误差均值
    made = 1.843 * mad
    return median, made

def imbalanced(top_k_indices, delta_threshold, data, X_copy_repair_indices):
    imbalanced_tuple_indices = set()

    # 初始化MinMaxScaler
    scaler_new = MinMaxScaler()
    data_imbalance = data
    # _, file_extension = os.path.splitext(file_path)
    # # 判断文件扩展名
    # if file_extension.lower() == '.xlsx':
    #     data_imbalance = pd.read_excel(file_path)
    # else:
    #     data_imbalance = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=False)

    feature_names = data_imbalance.columns.values.tolist()

    # if len(data_imbalance) > 20000:
    #     data_imbalance = data_imbalance.sample(n=20000, random_state=42)

    # 检测非数值列
    non_numeric_columns = data_imbalance.select_dtypes(exclude=[np.number]).columns

    # 为每个非数值列创建一个 LabelEncoder 实例
    encoders = {}
    for column in non_numeric_columns:
        encoder = LabelEncoder()
        data_imbalance[column] = encoder.fit_transform(data_imbalance[column])
        encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码

    data_imbalance[data_imbalance.columns] = scaler_new.fit_transform(data_imbalance[data_imbalance.columns])

    for feature in top_k_indices:
        select_feature = feature_names[feature]
        # 对每列数据进行分组
        bins = np.arange(0, 1.01, delta_threshold)  # 生成0-1之间100个间隔的数组
        digitized = np.digitize(data_imbalance[select_feature], bins)
        # 统计每个区间的计数
        unique_bins, counts = np.unique(digitized, return_counts=True)
        # 设置最小支持数差值
        median_imbalance, made_imbalance = calculate_made(counts)

        for t in X_copy_repair_indices:
            train_row_number = X_train.shape[0]
            ta = data_imbalance.iloc[t, feature]
            # 找到 ta 所在的间隔
            ta_bin = np.digitize([ta], bins)[0]
            # 找到 ta 所在间隔的计数
            ta_count = counts[unique_bins == ta_bin][0]
            lower_threshold = median_imbalance - 2 * made_imbalance
            upper_threshold = median_imbalance + 2 * made_imbalance
            if ta_count < lower_threshold or ta_count > upper_threshold:
                imbalanced_tuple_indices.add(t)

    X_copy_repair_imbalanced_indices = list(imbalanced_tuple_indices)

    return X_copy_repair_imbalanced_indices


def calculate_accuracy(ugly_outlier_index, true_ugly_indices, data_length):
    # 初始化所有索引为0 (表示不是丑陋离群值)
    y_pred = np.zeros(data_length, dtype=int)
    y_true = np.zeros(data_length, dtype=int)
    # ugly_outlier_index = np.array(list(ugly_outlier_index), dtype=int)
    # true_ugly_indices = np.array(true_ugly_indices, dtype=int)

    # 将预测的丑陋离群值标记为1
    y_pred[ugly_outlier_index] = 1
    # 将真实的丑陋离群值标记为1
    y_true[true_ugly_indices] = 1

    # 计算准确度
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# section 贝叶斯超参数优化
def objective(trial):
    # need_MDetI = trial.suggest_int("need_MDetI", low=0, high=1)
    # need_MDetO = trial.suggest_int("need_MDetO", low=0, high=1)
    # need_outlier = trial.suggest_int("need_outlier", low=0, high=1)
    # need_loss = trial.suggest_int("need_loss", low=0, high=1)
    # need_imbalanced = trial.suggest_int("need_imbalanced", low=0, high=1)
    need_MDetI = 1
    # need_MDetO = trial.suggest_int("need_MDetO", low=0, high=1)
    need_MDetO = 1
    # need_outlier = trial.suggest_int("need_outlier", low=0, high=1)
    need_outlier = 1
    # need_loss = trial.suggest_int("need_loss", low=0, high=1)
    need_loss = 1
    # need_imbalanced = trial.suggest_int("need_imbalanced", low=0, high=1)
    need_imbalanced = 1
    loss_choice = trial.suggest_categorical("loss_choice", ["cross_entropy"])

    # # theta=3
    # outlier_detector = trial.suggest_categorical("outlier_detector", ["GOAD", "RCA", "SLAD"])
    # theta_threshold = trial.suggest_float("theta_threshold", low=0.001, high=0.009, step=0.004)
    # delta_threshold = trial.suggest_float("delta_threshold", low=0.001, high=0.009, step=0.004)

    # # theta=4
    # outlier_detector = trial.suggest_categorical("outlier_detector", ["GOAD", "RCA", "SLAD", "DeepSVDD"])
    # theta_threshold = trial.suggest_float("theta_threshold", low=0.001, high=0.01, step=0.003)
    # delta_threshold = trial.suggest_float("delta_threshold", low=0.001, high=0.01, step=0.003)

    # # theta=5
    # outlier_detector = trial.suggest_categorical("outlier_detector", ["GOAD", "RCA", "SLAD", "DeepSVDD", "RePEN"])
    # theta_threshold = trial.suggest_float("theta_threshold", low=0.001, high=0.009, step=0.002)
    # delta_threshold = trial.suggest_float("delta_threshold", low=0.001, high=0.009, step=0.002)

    # # theta=6
    # outlier_detector = trial.suggest_categorical("outlier_detector", ["GOAD", "RCA", "SLAD", "DeepSVDD", "RePEN", "ICL"])
    # theta_threshold = trial.suggest_float("theta_threshold", low=0.001, high=0.011, step=0.002)
    # delta_threshold = trial.suggest_float("delta_threshold", low=0.001, high=0.011, step=0.002)

    # theta=7
    outlier_detector = trial.suggest_categorical("outlier_detector", ["GOAD", "RCA", "SLAD", "DeepSVDD", "RePEN", "ICL", "NeuTraL"])
    theta_threshold = trial.suggest_float("theta_threshold", low=0.001, high=0.013, step=0.002)
    delta_threshold = trial.suggest_float("delta_threshold", low=0.001, high=0.013, step=0.002)

    if need_MDetO == 1:
        t_start_MDetO = time.time()
        _, _, outliers_index_noise = MDetO(outlier_detector, X_train_copy, X_test_copy, X_copy)
        t_end_MDetO = time.time()
        t_MDetO = t_end_MDetO - t_start_MDetO
    else:
        outliers_index_noise = list(range(len(data_copy)))
        t_MDetO = 0
    if need_MDetI == 1:
        t_start_MDetI = time.time()
        top_k_indices = MDetI(svm_model_noise, data_copy, X_train_copy)
        t_end_MDetI = time.time()
        t_MDetI = t_end_MDetI - t_start_MDetI
    else:
        top_k_indices = list(range(len(data_copy.columns) - 1))
        t_MDetI = 0
    if need_loss == 1:
        ugly_outlier_candidates = loss(svm_model_noise, X_copy, X_test_copy, y_train, loss_choice)
    else:
        ugly_outlier_candidates = np.array(range(len(data_copy)))
    if need_outlier ==1:
        t_start_outlier = time.time()
        X_copy_repair_indices = outlier(data_copy, theta_threshold, top_k_indices, ugly_outlier_candidates)
        t_end_outlier = time.time()
        t_outlier = t_end_outlier - t_start_outlier
    else:
        X_copy_repair_indices = list(range(len(data_copy)))
        t_outlier = 0
    if need_imbalanced == 1:
        t_start_imbalanced = time.time()
        X_copy_repair_imbalanced_indices = imbalanced(top_k_indices, delta_threshold, data_copy.copy(), X_copy_repair_indices)
        t_end_imbalanced = time.time()
        t_imbalanced = t_end_imbalanced - t_start_imbalanced
    else:
        X_copy_repair_imbalanced_indices = list(range(len(data_copy)))
        t_imbalanced = 0
    if need_imbalanced == 1:
        ugly_outlier_index = np.union1d(outliers_index_noise, X_copy_repair_imbalanced_indices)
    else:
        ugly_outlier_index = np.union1d(outliers_index_noise, X_copy_repair_imbalanced_indices)

    # ugly_outlier_index = np.array(set(outliers_index_noise) & set(X_copy_repair_indices) & set(X_copy_repair_imbalanced_indices))
    t_start_predicate = time.time()
    y_pred = svm_model_noise.predict(X_copy)
    t_end_predicate = time.time()
    t_predicate = t_end_predicate - t_start_predicate
    true_ugly_indices = np.where(y_pred != y)[0]
    accuracy_noise = calculate_accuracy(ugly_outlier_index, true_ugly_indices, len(data_copy))
    t_train = t_MDetO + t_MDetI + t_outlier + t_imbalanced + t_predicate
    print("训练时间为：", t_train)
    return accuracy_noise

file_path = "./ugly_detection.log"
lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)

storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj),
)
# Execute an optimization by using an `Objective` instance.
study = optuna.create_study(storage=storage, direction="maximize")

t0 = time.time()  # 开始时间
study.optimize(objective, n_trials=1)
t1 = time.time()  # 结束时间
print("超参数优化耗时(含训练时间)：", t1-t0)

trial = study.best_trial
print("Objective Values: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


