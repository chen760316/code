"""
无监督离群值检测算法对ugly outliers的检测能力
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理

# subsection 原始真实数据集（对应实验测试1.1）

# file_path = "../../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../../datasets/real_outlier/annthyroid.csv"
# file_path = "../../datasets/real_outlier/optdigits.csv"
# file_path = "../../datasets/real_outlier/PageBlocks.csv"
# file_path = "../../datasets/real_outlier/pendigits.csv"
# file_path = "../../datasets/real_outlier/satellite.csv"
# file_path = "../../datasets/real_outlier/shuttle.csv"
# file_path = "../../datasets/real_outlier/yeast.csv"

# subsection 含有不同异常比例的真实数据集（对应实验测试1.2）

# choice Annthyroid数据集
# file_path = "../../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_02_v01.csv"
# file_path = "../../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_05_v01.csv"
# file_path = "../../datasets/real_outlier_varying_ratios/Annthyroid/Annthyroid_07.csv"

# choice Cardiotocography数据集
# file_path = "../../datasets/synthetic_outlier/Cardiotocography_local_0.1.csv"
# file_path = "../../datasets/synthetic_outlier/Cardiotocography_local_0.15.csv"
# file_path = "../../datasets/synthetic_outlier/Cardiotocography_local_0.2.csv"
# file_path = "../../datasets/synthetic_outlier/Cardiotocography_local_0.25.csv"
file_path = "../../datasets/synthetic_outlier/Cardiotocography_local_0.3.csv"

# choice PageBlocks数据集
# file_path = "../../datasets/real_outlier_varying_ratios/PageBlocks/PageBlocks_02_v01.csv"
# file_path = "../../datasets/real_outlier_varying_ratios/PageBlocks/PageBlocks_05_v01.csv"

# choice Wilt数据集
# file_path = "../../datasets/real_outlier_varying_ratios/Wilt/Wilt_02_v01.csv"
# file_path = "../../datasets/real_outlier_varying_ratios/Wilt/Wilt_05.csv"

# subsection 含有不同异常类型和异常比例的合成数据集（从真实数据中加入不同异常类型合成）（对应实验测试1.2）

# choice Annthyroid数据集+cluster噪声+不同噪声比例(效果稳定)
# file_path = "../../datasets/synthetic_outlier/annthyroid_cluster_0.1.csv"
# file_path = "../../datasets/synthetic_outlier/annthyroid_cluster_0.2.csv"
# file_path = "../../datasets/synthetic_outlier/annthyroid_cluster_0.3.csv"

# choice Cardiotocography数据集+local噪声+不同噪声比例(好用)
# file_path = "../../datasets/synthetic_outlier/Cardiotocography_local_0.1.csv"
# file_path = "../../datasets/synthetic_outlier/Cardiotocography_local_0.2.csv"
# file_path = "../../datasets/synthetic_outlier/Cardiotocography_local_0.3.csv"

# choice PageBlocks数据集+global噪声+不同噪声比例(效果稳定)
# file_path = "../../datasets/synthetic_outlier/PageBlocks_global_0.1.csv"
# file_path = "../../datasets/synthetic_outlier/PageBlocks_global_0.2.csv"
# file_path = "../../datasets/synthetic_outlier/PageBlocks_global_0.3.csv"

# choice satellite数据集+local噪声+不同噪声比例(好用)
# file_path = "../../datasets/synthetic_outlier/satellite_0.1.csv"
# file_path = "../../datasets/synthetic_outlier/satellite_0.2.csv"
# file_path = "../../datasets/synthetic_outlier/satellite_0.3.csv"

# choice annthyroid数据集+local噪声+不同噪声比例(好用)
# file_path = "../../datasets/synthetic_outlier/annthyroid_0.1.csv"
# file_path = "../../datasets/synthetic_outlier/annthyroid_0.2.csv"
# file_path = "../../datasets/synthetic_outlier/annthyroid_0.3.csv"

# choice waveform数据集+dependency噪声+不同噪声比例
# file_path = "../../datasets/synthetic_outlier/waveform_dependency_0.1.csv"
# file_path = "../../datasets/synthetic_outlier/waveform_dependency_0.2.csv"
# file_path = "../../datasets/synthetic_outlier/waveform_dependency_0.3.csv"

paths = [
    # "../../datasets/synthetic_outlier/Cardiotocography_local_0.1.csv",
    # "../../datasets/synthetic_outlier/Cardiotocography_local_0.15.csv",
    # "../../datasets/synthetic_outlier/Cardiotocography_local_0.2.csv",
    # "../../datasets/synthetic_outlier/Cardiotocography_local_0.25.csv",
    # "../../datasets/synthetic_outlier/Cardiotocography_local_0.3.csv"
    # "../../datasets/synthetic_outlier/PageBlocks_global_0.1.csv",
    # "../../datasets/synthetic_outlier/PageBlocks_global_0.15.csv",
    # "../../datasets/synthetic_outlier/PageBlocks_global_0.2.csv",
    # "../../datasets/synthetic_outlier/PageBlocks_global_0.25.csv",
    # "../../datasets/synthetic_outlier/PageBlocks_global_0.3.csv",
    # "../../datasets/synthetic_outlier/annthyroid_cluster_0.1.csv",
    # "../../datasets/synthetic_outlier/annthyroid_cluster_0.15.csv",
    # "../../datasets/synthetic_outlier/annthyroid_cluster_0.2.csv",
    # "../../datasets/synthetic_outlier/annthyroid_cluster_0.25.csv",
    # "../../datasets/synthetic_outlier/annthyroid_cluster_0.3.csv"
    "../../datasets/synthetic_outlier/waveform_dependency_0.1.csv",
    "../../datasets/synthetic_outlier/waveform_dependency_0.15.csv",
    "../../datasets/synthetic_outlier/waveform_dependency_0.2.csv",
    "../../datasets/synthetic_outlier/waveform_dependency_0.25.csv",
    "../../datasets/synthetic_outlier/waveform_dependency_0.3.csv"
]

for file_path in paths:
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

    # choice GOAD异常检测器
    # out_clf = GOAD(epochs=epochs, device=device, n_trans=n_trans)
    # out_clf.fit(X_train, y=None)
    # out_clf_noise = GOAD(epochs=epochs, device=device, n_trans=n_trans)
    # out_clf_noise.fit(X_train_copy, y=None)

    # choice DeepSVDD异常检测器
    # out_clf = DeepSVDD(epochs=epochs, device=device, random_state=random_state)
    # out_clf.fit(X_train, y=None)
    # out_clf_noise = DeepSVDD(epochs=epochs, device=device, random_state=random_state)
    # out_clf_noise.fit(X_train_copy, y=None)

    # choice RCA异常检测器
    # out_clf = RCA(epochs=epochs, device=device, act='LeakyReLU')
    # out_clf.fit(X_train)
    # out_clf_noise = RCA(epochs=epochs, device=device, act='LeakyReLU')
    # out_clf_noise.fit(X_train_copy)

    # choice RePEN异常检测器
    # out_clf = REPEN(epochs=5, device=device)
    # out_clf.fit(X_train)
    # out_clf_noise = REPEN(epochs=5, device=device)
    # out_clf_noise.fit(X_train_copy)

    # choice SLAD异常检测器
    # out_clf = SLAD(epochs=2, device=device)
    # out_clf.fit(X_train)
    # out_clf_noise = SLAD(epochs=2, device=device)
    # out_clf_noise.fit(X_train_copy)

    # choice ICL异常检测器
    out_clf = ICL(epochs=1, device=device, n_ensemble='auto')
    out_clf.fit(X_train)
    out_clf_noise = ICL(epochs=1, device=device, n_ensemble='auto')
    out_clf_noise.fit(X_train_copy)

    # choice NeuTraL异常检测器
    # out_clf = NeuTraL(epochs=1, device=device)
    # out_clf.fit(X_train)
    # out_clf_noise = NeuTraL(epochs=1, device=device)
    # out_clf_noise.fit(X_train_copy)

    # SECTION 在原始训练集和测试集上检测异常值

    # subsection 从原始训练集中检测出异常值索引

    print("*" * 100)
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
    print("训练集中的异常值比例：", len(train_outliers_index) / len(X_train))

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
    print("测试集中的异常值比例：", len(test_outliers_index) / len(X_test))

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
    print("加噪训练集中的异常值比例：", len(train_outliers_index_noise) / len(X_train_copy))

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
    print("加噪测试集中的异常值比例：", len(test_outliers_index_noise) / len(X_test_copy))

    # subsection 从全部加噪数据中检测出异常值索引

    print("*" * 100)
    scores_noise = out_clf_noise.decision_function(X_copy)
    pred_labels_noise, confidence_noise = out_clf_noise.predict(X_copy, return_confidence=True)
    outliers_index_noise = []
    for i in range(len(X_copy)):
        if pred_labels_noise[i] == 1:
            outliers_index_noise.append(i)
    print("加噪数据中的异常值数量：", len(outliers_index_noise))

    # section 训练下游任务的SVM模型

    # subsection 原始数据集上训练的SVM模型在训练集和测试集中分错的样本比例

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
                               cv=2,
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

    # 训练样本中被SVM模型错误分类的样本
    wrong_classified_train_indices = np.where(y_train != xgb_model.predict(X_train))[0]
    print("训练样本中被SVM模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices) / len(y_train))

    # 测试样本中被SVM模型错误分类的样本
    wrong_classified_test_indices = np.where(y_test != xgb_model.predict(X_test))[0]
    print("测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices) / len(y_test))

    # 整体数据集D中被SVM模型错误分类的样本
    print("完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices) + len(wrong_classified_test_indices)) / (len(y_train) + len(y_test)))

    # subsection 加噪数据集上训练的SVM模型在训练集和测试集中分错的样本比例

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
                               cv=2,
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

    # 加噪训练样本中被SVM模型错误分类的样本
    wrong_classified_train_indices_noise = np.where(y_train != xgb_model_noise.predict(X_train_copy))[0]
    print("加噪训练样本中被SVM模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise) / len(y_train))

    # 加噪测试样本中被SVM模型错误分类的样本
    wrong_classified_test_indices_noise = np.where(y_test != xgb_model_noise.predict(X_test_copy))[0]
    print("加噪测试样本中被SVM模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise) / len(y_test))

    # 整体加噪数据集D中被SVM模型错误分类的样本
    print("完整数据集D中被SVM模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise)) / (len(y_train) + len(y_test)))

    # section 计算加噪数据中的交叉熵损失

    from sklearn.preprocessing import OneHotEncoder
    from scipy.special import softmax

    y_mask = y.copy()
    y_mask[test_indices] = xgb_model_noise.predict(X_test_copy)
    # 对y_ground进行独热编码
    encoder = OneHotEncoder(sparse_output=False)
    y_true = encoder.fit_transform(y.reshape(-1, 1))

    # 获取每棵树的预测概率 (n_samples, n_classes)
    probabilities = xgb_model_noise.predict_proba(X_copy)

    # 假设我们需要对每个样本在所有类别上的分数应用 Softmax
    # 对于每个样本，随机森林会输出多个类别的概率，reshape 后应用 Softmax
    y_pred = softmax(probabilities, axis=1)

    # 计算每个样本的损失
    loss_per_sample = -np.sum(y_true * np.log(y_pred + 1e-12), axis=1)
    # 使用模型进行预测
    y_pred_copy = xgb_model_noise.predict(X_copy)
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
    # section 全部加噪数据中被SVM分类器误分类的数量
    label_pred = xgb_model_noise.predict(X_copy)
    wrong_classify_indices = []
    for i in range(len(X_copy)):
        if y[i] != label_pred[i]:
            wrong_classify_indices.append(i)
    print("被误分类的样本数量：", len(wrong_classify_indices))

    # section 检测ugly outliers的召回率
    ugly_found_by_detector = list(set(outliers_index_noise) & set(wrong_classify_indices))
    print("召回的ugly outliers的数量：", len(ugly_found_by_detector))
    print("ugly outliers的召回率为：", len(ugly_found_by_detector) / len(wrong_classify_indices))

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