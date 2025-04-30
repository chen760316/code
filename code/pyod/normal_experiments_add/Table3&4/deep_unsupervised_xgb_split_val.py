"""
无监督离群值检测算法对ugly outliers的检测能力
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.sod import SOD
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

file_path = "../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../datasets/real_outlier/annthyroid.csv"
# file_path = "../datasets/real_outlier/optdigits.csv"
# file_path = "../datasets/real_outlier/PageBlocks.csv"
# file_path = "../datasets/real_outlier/pendigits.csv"
# file_path = "../datasets/real_outlier/satellite.csv"
# file_path = "../datasets/real_outlier/shuttle.csv"
# file_path = "../datasets/real_outlier/yeast.csv"

def run(file_path):
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

    # section 数据特征缩放以及添加噪声

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

    # SECTION 选择无监督异常检测器

    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_trans = 64
    random_state = 42

    # choice ABOD异常检测器
    # out_clf = ABOD()
    # out_clf.fit(X_train)
    # out_clf_noise = ABOD()
    # out_clf_noise.fit(X_train_copy)

    # # choice COF异常检测器
    # out_clf = COF()
    # out_clf.fit(X_train)
    # out_clf_noise = COF()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice COPOD异常检测器
    # out_clf = COPOD()
    # out_clf.fit(X_train)
    # out_clf_noise = COPOD()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice ECOD异常检测器
    out_clf = ECOD()
    out_clf.fit(X_train)
    out_clf_noise = ECOD()
    out_clf_noise.fit(X_train_copy)
    #
    # # choice IForest异常检测器
    # out_clf = IForest()
    # out_clf.fit(X_train)
    # out_clf_noise = IForest()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice LODA异常检测器
    # out_clf = LODA()
    # out_clf.fit(X_train)
    # out_clf_noise = LODA()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice LOF异常检测器
    # out_clf = LOF()
    # out_clf.fit(X_train)
    # out_clf_noise = LOF()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice OCSVM异常检测器
    # out_clf = OCSVM()
    # out_clf.fit(X_train)
    # out_clf_noise = OCSVM()
    # out_clf_noise.fit(X_train_copy)
    #
    # # choice SOD异常检测器
    # out_clf = SOD()
    # out_clf.fit(X_train)
    # out_clf_noise = SOD()
    # out_clf_noise.fit(X_train_copy)

    # SECTION 在原始训练集和测试集上检测异常值

    # subsection 从原始训练集中检测出异常值索引

    print("*"*100)
    train_scores = out_clf.decision_function(X_train)
    train_pred_labels, train_confidence = out_clf.predict(X_train, return_confidence=True)
    print("训练集中异常值判定阈值为：", out_clf.threshold_)
    train_outliers_index = []
    print("训练集样本数：", len(X_train))
    for i in range(len(X_train)):
        if train_pred_labels[i] == 1:
            train_outliers_index.append(i)
    # 训练样本中的异常值索引
    print("训练集中异常值索引：", train_outliers_index)
    print("训练集中的异常值数量：", len(train_outliers_index))
    print("训练集中的异常值比例：", len(train_outliers_index)/len(X_train))

    # subsection 从原始测试集中检测出异常值索引

    test_scores = out_clf.decision_function(X_test)
    test_pred_labels, test_confidence = out_clf.predict(X_test, return_confidence=True)
    print("测试集中异常值判定阈值为：", out_clf.threshold_)
    test_outliers_index = []
    print("测试集样本数：", len(X_test))
    for i in range(len(X_test)):
        if test_pred_labels[i] == 1:
            test_outliers_index.append(i)
    # 训练样本中的异常值索引
    print("测试集中异常值索引：", test_outliers_index)
    print("测试集中的异常值数量：", len(test_outliers_index))
    print("测试集中的异常值比例：", len(test_outliers_index)/len(X_test))

    # section 从加噪数据集的训练集和测试集中检测出出异常值

    # subsection 从加噪训练集中检测出异常值索引

    train_scores_noise = out_clf_noise.decision_function(X_train_copy)
    train_pred_labels_noise, train_confidence_noise = out_clf_noise.predict(X_train_copy, return_confidence=True)
    print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
    train_outliers_index_noise = []
    print("加噪训练集样本数：", len(X_train_copy))
    for i in range(len(X_train_copy)):
        if train_pred_labels_noise[i] == 1:
            train_outliers_index_noise.append(i)
    # 训练样本中的异常值索引
    print("加噪训练集中异常值索引：", train_outliers_index_noise)
    print("加噪训练集中的异常值数量：", len(train_outliers_index_noise))
    print("加噪训练集中的异常值比例：", len(train_outliers_index_noise)/len(X_train_copy))

    # subsection 从加噪测试集中检测出异常值索引

    test_scores_noise = out_clf_noise.decision_function(X_test_copy)
    test_pred_labels_noise, test_confidence_noise = out_clf_noise.predict(X_test_copy, return_confidence=True)
    print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
    test_outliers_index_noise = []
    print("加噪测试集样本数：", len(X_test_copy))
    for i in range(len(X_test_copy)):
        if test_pred_labels_noise[i] == 1:
            test_outliers_index_noise.append(i)
    # 训练样本中的异常值索引
    print("加噪测试集中异常值索引：", test_outliers_index_noise)
    print("加噪测试集中的异常值数量：", len(test_outliers_index_noise))
    print("加噪测试集中的异常值比例：", len(test_outliers_index_noise)/len(X_test_copy))

    # subsection 从全部加噪数据中检测出异常值索引

    print("*"*100)
    scores_noise = out_clf_noise.decision_function(X_copy)
    pred_labels_noise, confidence_noise = out_clf_noise.predict(X_copy, return_confidence=True)
    outliers_index_noise = []
    for i in range(len(X_copy)):
        if pred_labels_noise[i] == 1:
            outliers_index_noise.append(i)
    print("加噪数据中的异常值数量：", len(outliers_index_noise))

    # section 训练下游任务的random forest模型

    # subsection 原始数据集上训练的random forest模型在训练集和测试集中分错的样本比例

    print("*" * 100)
    # 定义超参数搜索空间
    param_grid = {
        'n_estimators': [50, 100, 150],  # 树的数量
        'max_depth': [3, 6, 9],  # 树的最大深度
        'learning_rate': [0.01, 0.1, 0.2],  # 学习率
        'subsample': [0.8, 1.0],  # 子样本比例
        'colsample_bytree': [0.8, 1.0],  # 每棵树的特征采样比例
        'scale_pos_weight': [1, 2],  # 类别不平衡时调节
        'class_weight': [None]  # 兼容原格式，XGBoost 实际不支持 class_weight（占位）
    }

    # 初始化 XGBoost 模型（不用指定 n_estimators 之类的，这些会由 GridSearch 决定）
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        booster='gbtree',
        random_state=42,
        verbosity=0
    )

    # 初始化 GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring='accuracy',
                               cv=3,
                               verbose=2,
                               n_jobs=-1)

    # 拟合模型
    grid_search.fit(X_val, y_val)

    # 输出最佳超参数和交叉验证得分
    print("最佳超参数组合：", grid_search.best_params_)
    print("最佳交叉验证准确度：", grid_search.best_score_)

    # 获取最佳模型
    xgb_model = grid_search.best_estimator_
    xgb_model.fit(X_train, y_train)
    train_label_pred = xgb_model.predict(X_train)

    # 训练样本中被random forest模型错误分类的样本
    wrong_classified_train_indices = np.where(y_train != xgb_model.predict(X_train))[0]
    print("训练样本中被random forest模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))

    # 测试样本中被random forest模型错误分类的样本
    wrong_classified_test_indices = np.where(y_test != xgb_model.predict(X_test))[0]
    print("测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))

    # 整体数据集D中被random forest模型错误分类的样本
    print("完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

    # subsection 加噪数据集上训练的random forest模型在训练集和测试集中分错的样本比例

    print("*" * 100)
    # 定义超参数搜索空间
    param_grid = {
        'n_estimators': [50, 100, 150],  # 树的数量
        'max_depth': [3, 6, 9],  # 树的最大深度
        'learning_rate': [0.01, 0.1, 0.2],  # 学习率
        'subsample': [0.8, 1.0],  # 子样本比例
        'colsample_bytree': [0.8, 1.0],  # 每棵树的特征采样比例
        'scale_pos_weight': [1, 2],  # 类别不平衡时调节
        'class_weight': [None]  # 兼容原格式，XGBoost 实际不支持 class_weight（占位）
    }

    # 初始化 XGBoost 模型（不用指定 n_estimators 之类的，这些会由 GridSearch 决定）
    xgb_model_noise = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        booster='gbtree',
        random_state=42,
        verbosity=0
    )

    # 初始化 GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model_noise, param_grid=param_grid,
                               scoring='accuracy',
                               cv=3,
                               verbose=2,
                               n_jobs=-1)

    # 拟合模型
    grid_search.fit(X_val, y_val)

    # 输出最佳超参数和交叉验证得分
    print("最佳超参数组合：", grid_search.best_params_)
    print("最佳交叉验证准确度：", grid_search.best_score_)

    # 获取最佳模型
    xgb_model_noise = grid_search.best_estimator_

    xgb_model_noise.fit(X_train_copy, y_train)
    train_label_pred_noise = xgb_model_noise.predict(X_train_copy)

    # 加噪训练样本中被random forest模型错误分类的样本
    wrong_classified_train_indices_noise = np.where(y_train != xgb_model_noise.predict(X_train_copy))[0]
    print("加噪训练样本中被random forest模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise)/len(y_train))

    # 加噪测试样本中被random forest模型错误分类的样本
    wrong_classified_test_indices_noise = np.where(y_test != xgb_model_noise.predict(X_test_copy))[0]
    print("加噪测试样本中被random forest模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise)/len(y_test))

    # 整体加噪数据集D中被random forest模型错误分类的样本
    print("完整数据集D中被random forest模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise))/(len(y_train) + len(y_test)))

    # section 计算加噪数据中的交叉熵损失

    # 获取概率值
    y_pred = xgb_model.predict_proba(X_copy)[:, [1, 0]]
    # 创建 OneHotEncoder 实例
    encoder = OneHotEncoder(sparse_output=False)
    # 拟合并转换 y_test
    y_true = encoder.fit_transform(y.reshape(-1, 1))
    # 计算每个样本的损失
    loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
    # 计算测试集平均多分类交叉熵损失
    average_loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
    bad_samples_noise = np.where(loss_per_sample > average_loss)[0]
    good_samples_noise = np.where(loss_per_sample <= average_loss)[0]
    print("损失不超过阈值的样本数量：", len(good_samples_noise))
    print("损失大于阈值的样本数量：", len(bad_samples_noise))

    # section 全部加噪数据中被random forest分类器误分类的数量
    label_pred = xgb_model_noise.predict(X_copy)
    wrong_classify_indices = []
    for i in range(len(X_copy)):
        if y[i] != label_pred[i]:
            wrong_classify_indices.append(i)
    print("被误分类的样本数量：", len(wrong_classify_indices))

    # section 检测ugly outliers的召回率
    ugly_found_by_detector = list(set(outliers_index_noise) & set(wrong_classify_indices))
    print("召回的ugly outliers的数量：", len(ugly_found_by_detector))
    print("ugly outliers的召回率为：", len(ugly_found_by_detector)/len(wrong_classify_indices))

    # section 重新计算recall/precision/F1分数
    # 计算 TP, FN, FP
    TP = len(set(ugly_found_by_detector) & set(wrong_classify_indices))  # 交集元素数量
    FN = len(set(wrong_classify_indices) - set(ugly_found_by_detector))  # s2中有但s1中没有的元素数量
    FP = len(set(ugly_found_by_detector) - set(wrong_classify_indices))  # s1中有但s2中没有的元素数量

    # 计算召回率 (Recall)
    Recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # 计算精确度 (Precision)
    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    # 计算 F1 分数
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0

    # 打印结果
    print("*" * 100)
    print("候选的ugly outliers列表长度为：", len(ugly_found_by_detector))
    print("真实的ugly outliers列表长度为：", len(wrong_classify_indices))
    print(f"Recall: {Recall:.4f}")
    print(f"Precision: {Precision:.4f}")
    print(f"F1 Score: {F1:.4f}")
    return len(ugly_found_by_detector) / len(wrong_classify_indices)


if __name__ == '__main__':
    paths = [
        "../datasets/real_outlier/Cardiotocography.csv",
        "../datasets/real_outlier/annthyroid.csv",
        "../datasets/real_outlier/optdigits.csv",
        "../datasets/real_outlier/PageBlocks.csv",
        "../datasets/real_outlier/pendigits.csv",
        "../datasets/real_outlier/satellite.csv",
        "../datasets/real_outlier/shuttle.csv",
        "../datasets/real_outlier/yeast.csv"
    ]
    res_list = []
    for file_path in paths:
        res = run(file_path)
        res_list.append(res)
    print(res_list)