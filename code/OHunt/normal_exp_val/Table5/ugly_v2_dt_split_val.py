"""
𝑅(𝑡) ∧ M𝑜 (𝑡, D) ∧ 𝑋1 → ugly(𝑡)
采用不同修复策略对上规则形式发现的ugly outliers进行修复
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from deepod.models.tabular import GOAD
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from deepod.models.tabular import PReNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 标准数据集处理，输入原始多分类数据集，在中间处理过程转化为异常检测数据集

# choice drybean数据集(效果好)
# file_path = "../datasets/multi_class/drybean.xlsx"
# data = pd.read_excel(file_path)

# choice obesity数据集(效果好)
# file_path = "../datasets/multi_class/obesity.csv"
# data = pd.read_csv(file_path)

# choice balita数据集(SVM拟合效果差，但修复后效果提升显著)
# file_path = "../datasets/multi_class/balita.csv"
# data = pd.read_csv(file_path)

# choice apple数据集(效果提升小)
# file_path = "../datasets/multi_class/apple.csv"
# data = pd.read_csv(file_path)

# choice adult数据集(效果提升明显)
# file_path = "../datasets/multi_class/adult.csv"
# data = pd.read_csv(file_path)

# choice body数据集(执行较慢，SVM拟合效果很差，但修复后效果有提升)
# file_path = "../datasets/multi_class/body/body.csv"
# data = pd.read_csv(file_path)

# # choice covertype数据集(执行很慢，SVM拟合效果很差，但修复后效果提升显著)
# file_path = "../datasets/multi_class/covtype/covtype_process.csv"
# data = pd.read_csv(file_path)

# choice financial(SVM拟合效果很差，但修复后效果提升显著)
# file_path = "../datasets/multi_class/financial/financial.csv"
# data = pd.read_csv(file_path)

# choice online(SVM拟合效果相对较差，但修复后效果提升显著)
# file_path = "../datasets/multi_class/online/online.csv"
# data = pd.read_csv(file_path)

# choice star(SVM拟合相对准确，修复后效果提升显著)
# file_path = "../datasets/multi_class/star/star.csv"
# data = pd.read_csv(file_path)

# choice student(SVM拟合相对较差，修复后效果提升显著)
# file_path = "../datasets/multi_class/student/Student.csv"
# data = pd.read_csv(file_path)

# choice Iris数据集(效果一般)
# file_path = "../datasets/multi_class/Iris.csv"
# data = pd.read_csv(file_path)

# choice 真实异常检测数据集+local类型异常（需要搭配非线性SVM，线性SVM下无法很好划分）
# file_path = "../datasets/synthetic_outlier/annthyroid_0.1.csv"
# data = pd.read_csv(file_path)

# choice wine数据集(SVM拟合效果差，修复效果差)
# file_path = "../datasets/multi_class/wine.csv"
# data = pd.read_csv(file_path, sep=';')

# choice 真实异常检测数据集（本身不包含错误数据，不适合用于修复任务，且需要搭配非线性SVM）
# file_path = "../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../datasets/real_outlier/annthyroid.csv"
# file_path = "../datasets/real_outlier/optdigits.csv"
# file_path = "../datasets/real_outlier/PageBlocks.csv"
# file_path = "../datasets/real_outlier/pendigits.csv"
# file_path = "../datasets/real_outlier/satellite.csv"
# file_path = "../datasets/real_outlier/shuttle.csv"
# file_path = "../datasets/real_outlier/yeast.csv"
# data = pd.read_csv(file_path)

paths = [
    # "../datasets/multi_class/drybean.csv",
    # "../datasets/multi_class/adult.csv",
    "../datasets/multi_class/online/online.csv",
    "../datasets/multi_class/star/star.csv",
    "../datasets/multi_class/student/Student.csv",
    "../datasets/real_outlier/Cardiotocography.csv",
    # "../datasets/real_outlier/annthyroid.csv",
    # "../datasets/real_outlier/optdigits.csv",
    # "../datasets/real_outlier/PageBlocks.csv",
    # "../datasets/real_outlier/pendigits.csv",
    # "../datasets/real_outlier/satellite.csv",
    # "../datasets/real_outlier/shuttle.csv",
    "../datasets/real_outlier/yeast.csv",
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

    # 找到分类特征的列名
    categorical_columns = data.select_dtypes(exclude=['float']).columns[:-1]
    # 获取分类特征对应的索引
    categorical_features = [data.columns.get_loc(col) for col in categorical_columns]

    all_columns = data.columns.values.tolist()
    feature_names = all_columns[:-1]
    class_name = all_columns[-1]

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

    # section 数据特征缩放和数据加噪

    # 数据标准化（对所有特征进行标准化）
    X = StandardScaler().fit_transform(X)
    feature_names = data.columns.values.tolist()
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

    # SECTION M𝑜 (𝑡, D) 针对元组异常的无监督异常检测器PReNet
    # subsection 设置训练测试弱监督样本
    # 设置弱监督训练样本
    # 找到所有标签为 1 的样本索引
    semi_label_ratio = 1.0  # 设置已知的异常标签比例
    positive_indices = np.where(y_train == min_label)[0]
    # 随机选择 10% 的正样本
    n_positive_to_keep = int(len(positive_indices) * semi_label_ratio)
    selected_positive_indices = np.random.choice(positive_indices, n_positive_to_keep, replace=False)
    # 创建用于异常检测器的训练标签
    y_semi = np.zeros_like(y_train)  # 默认全为 0
    y_semi[selected_positive_indices] = 1  # 设置选中的正样本为 1
    # 创建用于异常检测器的测试标签
    y_semi_test = np.zeros_like(y_test)
    test_positive_indices = np.where(y_test == min_label)[0]
    y_semi_test[test_positive_indices] = 1

    # subsection 异常检测器训练
    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_trans = 64
    random_state = 42

    out_clf = PReNet(epochs=epochs, device=device, random_state=random_state)
    out_clf.fit(X_train, y=y_semi)

    out_clf_noise = PReNet(epochs=epochs, device=device, random_state=random_state)
    out_clf_noise.fit(X_train_copy, y_semi)

    # SECTION 借助异常检测器，在训练集上进行异常值检测
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
    train_correct_detect_samples = []
    for i in range(len(X_train)):
        if train_pred_labels[i] == y_semi[i]:
            train_correct_detect_samples.append(i)
    print("训练集中异常检测器的检测准确度：", len(train_correct_detect_samples) / len(X_train))
    # 训练样本中的异常值索引
    print("训练集中异常值索引：", train_outliers_index)
    print("训练集中的异常值数量：", len(train_outliers_index))
    print("训练集中的异常值比例：", len(train_outliers_index) / len(X_train))

    # subsection 从原始测试集中检测出异常值索引

    print("*" * 100)
    test_scores = out_clf.decision_function(X_test)
    test_pred_labels, test_confidence = out_clf.predict(X_test, return_confidence=True)
    print("测试集中异常值判定阈值为：", out_clf.threshold_)
    test_outliers_index = []
    print("测试集样本数：", len(X_test))
    for i in range(len(X_test)):
        if test_pred_labels[i] == 1:
            test_outliers_index.append(i)
    test_correct_detect_samples = []
    for i in range(len(X_test)):
        if test_pred_labels[i] == y_semi_test[i]:
            test_correct_detect_samples.append(i)
    print("测试集中异常检测器的检测准确度：", len(test_correct_detect_samples) / len(X_test))
    # 训练样本中的异常值索引
    print("测试集中异常值索引：", test_outliers_index)
    print("测试集中的异常值数量：", len(test_outliers_index))
    print("测试集中的异常值比例：", len(test_outliers_index) / len(X_test))

    # section 从加噪数据集的训练集和测试集中检测出的异常值
    # subsection 从加噪训练集中检测出异常值索引

    print("*" * 100)
    train_scores_noise = out_clf_noise.decision_function(X_train_copy)
    train_pred_labels_noise, train_confidence_noise = out_clf_noise.predict(X_train_copy, return_confidence=True)
    print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
    train_outliers_index_noise = []
    print("加噪训练集样本数：", len(X_train_copy))
    for i in range(len(X_train_copy)):
        if train_pred_labels_noise[i] == 1:
            train_outliers_index_noise.append(i)
    train_correct_detect_samples_noise = []
    for i in range(len(X_train_copy)):
        if train_pred_labels_noise[i] == y_semi[i]:
            train_correct_detect_samples_noise.append(i)
    print("训练集中异常检测器的检测准确度：", len(train_correct_detect_samples_noise) / len(X_train_copy))
    # 训练样本中的异常值索引
    print("加噪训练集中异常值索引：", train_outliers_index_noise)
    print("加噪训练集中的异常值数量：", len(train_outliers_index_noise))
    print("加噪训练集中的异常值比例：", len(train_outliers_index_noise) / len(X_train_copy))

    # subsection 从加噪测试集中检测出异常值索引

    print("*" * 100)
    test_scores_noise = out_clf_noise.decision_function(X_test_copy)
    test_pred_labels_noise, test_confidence_noise = out_clf_noise.predict(X_test_copy, return_confidence=True)
    print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
    test_outliers_index_noise = []
    print("加噪测试集样本数：", len(X_test_copy))
    for i in range(len(X_test_copy)):
        if test_pred_labels_noise[i] == 1:
            test_outliers_index_noise.append(i)
    test_correct_detect_samples_noise = []
    print(len(test_pred_labels_noise), len(y_test))
    for i in range(len(X_test_copy)):
        if test_pred_labels_noise[i] == y_semi_test[i]:
            test_correct_detect_samples_noise.append(i)
    print("测试集中异常检测器的检测准确度：", len(test_correct_detect_samples_noise) / len(X_test_copy))
    # 训练样本中的异常值索引
    print("加噪测试集中异常值索引：", test_outliers_index_noise)
    print("加噪测试集中的异常值数量：", len(test_outliers_index_noise))
    print("加噪测试集中的异常值比例：", len(test_outliers_index_noise) / len(X_test_copy))

    # SECTION decision tree模型的实现和准确度测试

    # subsection 原始数据集上训练的decision tree模型在训练集和测试集中分错的样本比例

    print("*" * 100)
    # 定义一个超参数网格
    param_grid = {
        'max_depth': [5, 10, 15, None],  # 树的最大深度
        'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 叶节点的最小样本数
        'max_features': ['auto', 'sqrt', 'log2', None],  # 最大特征数
        'class_weight': ['balanced', None],  # 类别权重
        'criterion': ['gini', 'entropy']  # 划分标准
    }

    # 初始化决策树模型
    dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)

    # 初始化GridSearchCV进行超参数调优，使用验证集进行交叉验证
    grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid,
                               cv=2,  # 3折交叉验证
                               scoring='accuracy',  # 使用准确度作为评分指标
                               n_jobs=-1,  # 使用所有CPU核进行并行计算
                               verbose=2)  # 打印详细的进度信息

    # 在验证集上进行网格搜索
    grid_search.fit(X_val, y_val)

    # 输出最佳参数和最佳交叉验证得分
    print("最佳超参数组合：", grid_search.best_params_)
    print("最佳交叉验证准确度：", grid_search.best_score_)

    # 使用最佳超参数训练模型
    dt_model = grid_search.best_estimator_
    dt_model.fit(X_train, y_train)
    train_label_pred = dt_model.predict(X_train)
    test_label_pred = dt_model.predict(X_test)

    # 训练样本中被decision tree模型错误分类的样本
    wrong_classified_train_indices = np.where(y_train != dt_model.predict(X_train))[0]
    print("训练样本中被decision tree模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices) / len(y_train))

    # 测试样本中被decision tree模型错误分类的样本
    wrong_classified_test_indices = np.where(y_test != dt_model.predict(X_test))[0]
    print("测试样本中被decision tree模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices) / len(y_test))

    # 整体数据集D中被decision tree模型错误分类的样本
    print("完整数据集D中被decision tree模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices) + len(wrong_classified_test_indices)) / (len(y_train) + len(y_test)))

    # subsection 加噪数据集上训练的decision tree模型在训练集和测试集中分错的样本比例

    print("*" * 100)
    # # 定义一个超参数网格
    # param_grid = {
    #     'max_depth': [5, 10, 15, None],  # 树的最大深度
    #     'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
    #     'min_samples_leaf': [1, 2, 4],  # 叶节点的最小样本数
    #     'max_features': ['auto', 'sqrt', 'log2', None],  # 最大特征数
    #     'class_weight': ['balanced', None],  # 类别权重
    #     'criterion': ['gini', 'entropy']  # 划分标准
    # }
    #
    # # 初始化决策树模型
    # dt_model_noise = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    #
    # # 初始化GridSearchCV进行超参数调优，使用验证集进行交叉验证
    # grid_search = GridSearchCV(estimator=dt_model_noise, param_grid=param_grid,
    #                            cv=2,  # 3折交叉验证
    #                            scoring='accuracy',  # 使用准确度作为评分指标
    #                            n_jobs=-1,  # 使用所有CPU核进行并行计算
    #                            verbose=2)  # 打印详细的进度信息
    #
    # # 在验证集上进行网格搜索
    # grid_search.fit(X_val, y_val)
    #
    # # 输出最佳参数和最佳交叉验证得分
    # print("最佳超参数组合：", grid_search.best_params_)
    # print("最佳交叉验证准确度：", grid_search.best_score_)
    #
    # # 使用最佳超参数训练模型
    # dt_model_noise = grid_search.best_estimator_
    # dt_model_noise.fit(X_train_copy, y_train)
    dt_model_noise = dt_model
    train_label_pred_noise = dt_model_noise.predict(X_train_copy)
    test_label_pred_noise = dt_model_noise.predict(X_test_copy)

    # 加噪训练样本中被decision tree模型错误分类的样本
    wrong_classified_train_indices_noise = np.where(y_train != dt_model_noise.predict(X_train_copy))[0]
    print("加噪训练样本中被decision tree模型错误分类的样本占总加噪训练样本的比例：", len(wrong_classified_train_indices_noise) / len(y_train))

    # 加噪测试样本中被decision tree模型错误分类的样本
    wrong_classified_test_indices_noise = np.where(y_test != dt_model_noise.predict(X_test_copy))[0]
    print("加噪测试样本中被decision tree模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices_noise) / len(y_test))

    # 整体加噪数据集D中被decision tree模型错误分类的样本
    print("完整数据集D中被decision tree模型错误分类的样本占总完整数据的比例：", (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise)) / (len(y_train) + len(y_test)))

    # section 修复前实验指标测定

    """Accuracy指标"""
    print("*" * 100)
    print("分类器在修复前的加噪测试集中的分类准确度：" + str(accuracy_score(y_test, test_label_pred_noise)))

    """Precision/Recall/F1指标"""

    # average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
    # average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
    # average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
    # average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。

    print("分类器在修复前的加噪测试集中的分类精确度：" + str(precision_score(y_test, test_label_pred_noise, average='weighted')))
    print("分类器在修复前的加噪测试集中的分类召回率：" + str(recall_score(y_test, test_label_pred_noise, average='weighted')))
    print("分类器在修复前的加噪测试集中的分类F1分数：" + str(f1_score(y_test, test_label_pred_noise, average='weighted')))

    # section 确定有影响力的特征
    # choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)
    import re

    # 特征数取4或6
    i = len(feature_names)
    np.random.seed(1)
    categorical_names = {}
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data.iloc[:, feature])
        data.iloc[:, feature] = le.transform(data.iloc[:, feature])
        categorical_names[feature] = le.classes_
    explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_name,
                                     categorical_features=categorical_features,
                                     categorical_names=categorical_names, kernel_width=3)
    # predict_proba 方法用于分类任务，predict 方法用于回归任务
    predict_fn = lambda x: dt_model.predict_proba(x)
    exp = explainer.explain_instance(X_train[i], predict_fn, num_features=len(feature_names) // 2)
    # 获取最具影响力的特征及其权重
    top_features = exp.as_list()
    top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
    top_k_indices = [feature_names.index(name) for name in top_feature_names]
    print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

    # # 获取最具影响力的特征及其权重
    # top_features = exp.as_list()
    # important_features = []
    # for feature_set in top_features:
    #     feature_long = feature_set[0]
    #     for feature in feature_names:
    #         if set(feature).issubset(set(feature_long)):
    #             important_features.append(feature)
    #             break
    # top_k_indices = [feature_names.index(feature_name) for feature_name in important_features]
    # print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

    # section 识别X_copy中需要修复的元组

    # 异常检测器检测出的训练集和测试集中的异常值在原含噪数据D'中的索引
    train_outliers_noise = train_indices[train_outliers_index_noise]
    test_outliers_noise = test_indices[test_outliers_index_noise]
    outliers_noise = np.union1d(train_outliers_noise, test_outliers_noise)

    # 在加噪数据集D'上训练的decision tree模型，其分类错误的样本在原含噪数据D'中的索引
    train_wrong_clf_noise = train_indices[wrong_classified_train_indices_noise]
    test_wrong_clf_noise = test_indices[wrong_classified_test_indices_noise]
    wrong_clf_noise = np.union1d(train_wrong_clf_noise, test_wrong_clf_noise)

    # outliers和分错样本的并集
    train_union = np.union1d(train_outliers_noise, train_wrong_clf_noise)
    test_union = np.union1d(test_outliers_noise, test_wrong_clf_noise)

    # 加噪数据集D'上需要修复的值
    # 需要修复的特征和标签值
    X_copy_repair_indices = np.union1d(outliers_noise, wrong_clf_noise)


    # section 选取活动域过小的特征

    def calculate_made(data):
        median = np.median(data)  # 计算中位数
        abs_deviation = np.abs(data - median)  # 计算每个数据点与中位数的绝对误差
        mad = np.median(abs_deviation)  # 计算绝对误差均值
        made = 1.843 * mad
        return median, made


    # 初始化MinMaxScaler
    scaler = MinMaxScaler()
    if file_path == "../datasets/multi_class/drybean.xlsx":
        data_minmax = pd.read_excel(file_path)
    else:
        data_minmax = pd.read_csv(file_path)

    if len(data_minmax) > 20000:
        data_minmax = data_minmax.sample(n=20000, random_state=42)

    # 检测非数值列
    non_numeric_columns = data_minmax.select_dtypes(exclude=[np.number]).columns

    # 为每个非数值列创建一个 LabelEncoder 实例
    encoders = {}
    for column in non_numeric_columns:
        encoder = LabelEncoder()
        data_minmax[column] = encoder.fit_transform(data_minmax[column])
        encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码

    data_minmax[data_minmax.columns] = scaler.fit_transform(data_minmax[data_minmax.columns])
    # 设置分组的间隔
    interval = 0.01
    # 对每列数据进行分组
    bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
    columns_bins = {}
    columns_bins_count = []
    small_domain_features = []

    for column in data_minmax.columns:
        digitized = np.digitize(data_minmax[column], bins)
        unique_bins, counts = np.unique(digitized, return_counts=True)
        columns_bins[column] = len(unique_bins)
        columns_bins_count.append(len(unique_bins))

    for i in top_k_indices:
        select_feature = feature_names[i]
        selected_bins = columns_bins[select_feature]
        median, made = calculate_made(np.array(columns_bins_count))
        lower_threshold = median - 2 * made
        upper_threshold = median + 2 * made
        if selected_bins < lower_threshold:
            small_domain_features.append(i)
    filtered_important_feature_indices = [item for item in top_k_indices if item not in small_domain_features]

    imbalanced_tuple_indices = set()

    # 初始化MinMaxScaler
    scaler_new = MinMaxScaler()
    if file_path == "../datasets/multi_class/drybean.xlsx":
        data_imbalance = pd.read_excel(file_path)
    else:
        data_imbalance = pd.read_csv(file_path)

    if len(data_imbalance) > 20000:
        data_imbalance = data_imbalance.sample(n=20000, random_state=42)

    # 检测非数值列
    non_numeric_columns = data_imbalance.select_dtypes(exclude=[np.number]).columns

    # 为每个非数值列创建一个 LabelEncoder 实例
    encoders = {}
    for column in non_numeric_columns:
        encoder = LabelEncoder()
        data_imbalance[column] = encoder.fit_transform(data_imbalance[column])
        encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码

    data_imbalance[data_imbalance.columns] = scaler_new.fit_transform(data_imbalance[data_imbalance.columns])

    for feature in filtered_important_feature_indices:
        select_feature = feature_names[feature]
        # 对每列数据进行分组
        bins = np.arange(0, 1.01, interval)  # 生成0-1之间100个间隔的数组
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

    X_copy_repair_indices = list(imbalanced_tuple_indices)
    X_copy_repair = X_copy[X_copy_repair_indices]
    y_repair = y[X_copy_repair_indices]

    # 生成保留的行索引
    rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

    # 使用保留的行索引选择D'中的正常数据
    # 无需修复的特征和标签值
    X_copy_inners = X_copy[rows_to_keep]
    y_inners = y[rows_to_keep]

    # # section 方案一：对X_copy中需要修复的元组进行标签修复（knn方法）
    # #  需要修复的元组通过异常值检测器检测到的元组和decision tree分类错误的元组共同确定（取并集）
    #
    # # subsection 尝试修复异常数据的标签
    #
    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(X_copy_inners, y_inners)
    #
    # # 预测异常值
    # y_pred = knn.predict(X_copy_repair)
    #
    # # 替换异常值
    # y[X_copy_repair_indices] = y_pred
    # y_train = y[train_indices]
    # y_test = y[test_indices]
    #
    # # subsection 重新在修复后的数据上训练decision tree模型
    #
    # dt_repari = tree.DecisionTreeClassifier(class_weight='balanced')
    # dt_repari.fit(X_train_copy, y_train)
    # y_train_pred = dt_repari.predict(X_train_copy)
    # y_test_pred = dt_repari.predict(X_test_copy)

    # # section 方案二：对X_copy中需要修复的元组进行特征修复（统计方法修复）
    # #  需要修复的元组通过异常值检测器检测到的元组和decision tree分类错误的元组共同确定（取并集）(修复效果由于监督/无监督基准)
    #
    # # subsection 确定有影响力特征中的离群值并采用均值修复
    # for i in range(X_copy.shape[1]):
    #     if i in top_k_indices:
    #         column_data = X_copy[:, i]
    #         mean = np.mean(column_data)
    #         # 将所有需要修复的行对应的列位置的元素替换为均值
    #         intersection = X_copy_repair_indices
    #         X_copy[intersection, i] = mean
    #
    # X_train_copy = X_copy[train_indices]
    # X_test_copy = X_copy[test_indices]
    #
    # # subsection 重新在修复后的数据上训练decision tree模型
    #
    # dt_repari = tree.DecisionTreeClassifier(class_weight='balanced')
    # dt_repari.fit(X_train_copy, y_train)
    # y_train_pred = dt_repari.predict(X_train_copy)
    # y_test_pred = dt_repari.predict(X_test_copy)
    #
    # print("*" * 100)
    # # 训练样本中被decision tree模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    # print("加噪标签修复后，训练样本中被decision tree模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
    #
    # # 测试样本中被decision tree模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    # print("加噪标签修复后，测试样本中被decision tree模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
    #
    # # 整体数据集D中被decision tree模型错误分类的样本
    # print("加噪标签修复后，完整数据集D中被decision tree模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

    # # section 方案三：对X_copy中需要修复的元组借助knn进行修复，choice1 将异常元组中的元素直接设置为nan(修复误差太大，修复后准确性下降)
    # #  choice2 仅将有影响力特征上的元素设置为np.nan
    #
    # # # choice 将异常元组中的所有元素设置为nan
    # # for i in range(X_copy.shape[1]):
    # #     X_copy[X_copy_repair_indices, i] = np.nan
    #
    # # choice 仅将异常元组中的有影响力的元素设置为nan
    # for i in top_k_indices:
    #     X_copy[X_copy_repair_indices, i] = np.nan
    #
    # # choice 使用knn修复所有被标记为nan的异常特征
    # # 创建 KNN Imputer 对象
    # knn_imputer = KNNImputer(n_neighbors=5)
    #
    # # 使用 KNN 算法填补异常特征
    # X_copy = knn_imputer.fit_transform(X_copy)
    # X_train_copy = X_copy[train_indices]
    # X_test_copy = X_copy[test_indices]
    #
    # dt_repari = tree.DecisionTreeClassifier(class_weight='balanced')
    # dt_repari.fit(X_train_copy, y_train)
    # y_train_pred = dt_repari.predict(X_train_copy)
    # y_test_pred = dt_repari.predict(X_test_copy)
    #
    # print("*" * 100)
    # # 训练样本中被decision tree模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    # print("借助knn修复需要修复的样本后，训练样本中被decision tree模型错误分类的样本占总训练样本的比例：",
    #       len(wrong_classified_train_indices)/len(y_train))
    #
    # # 测试样本中被decision tree模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    # print("借助knn修复需要修复的样本后，测试样本中被decision tree模型错误分类的样本占总测试样本的比例：",
    #       len(wrong_classified_test_indices)/len(y_test))
    #
    # # 整体数据集D中被decision tree模型错误分类的样本
    # print("借助knn修复需要修复的样本后，完整数据集D中被decision tree模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))
    #       /(len(y_train) + len(y_test)))

    # # section 方案四：将X_copy中训练集和测试集需要修复的元组直接删除，在去除后的训练集上训练decision tree模型
    #
    # set_X_copy_repair = set(X_copy_repair_indices)
    #
    # # 计算差集，去除训练集中需要修复的的元素
    # set_train_indices = set(train_indices)
    # remaining_train_indices = list(set_train_indices - set_X_copy_repair)
    # X_train_copy_repair = X_copy[remaining_train_indices]
    # y_train_copy_repair = y[remaining_train_indices]
    #
    # # # choice 计算差集，去除测试集中需要修复的的元素
    # # set_test_indices = set(test_indices)
    # # remaining_test_indices = list(set_test_indices - set_X_copy_repair)
    # # X_test_copy_repair = X_copy[remaining_test_indices]
    # # y_test_copy_repair = y[remaining_test_indices]
    #
    # # choice 不删除测试集中的离群样本
    # X_test_copy_repair = X_copy[test_indices]
    # y_test_copy_repair = y[test_indices]
    #
    # # subsection 重新在修复后的数据上训练decision tree模型
    #
    # dt_repari = tree.DecisionTreeClassifier(class_weight='balanced')
    # dt_repari.fit(X_train_copy_repair, y_train_copy_repair)
    # y_train_pred = dt_repari.predict(X_train_copy_repair)
    # y_test_pred = dt_repari.predict(X_test_copy_repair)
    #
    # print("*" * 100)
    # # 训练样本中被decision tree模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train_copy_repair != y_train_pred)[0]
    # print("删除需要修复的样本后，训练样本中被decision tree模型错误分类的样本占总训练样本的比例：",
    #       len(wrong_classified_train_indices)/len(y_train_copy_repair))
    #
    # # 测试样本中被decision tree模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test_copy_repair != y_test_pred)[0]
    # print("删除需要修复的样本后，测试样本中被decision tree模型错误分类的样本占总测试样本的比例：",
    #       len(wrong_classified_test_indices)/len(y_test_copy_repair))
    #
    # # 整体数据集D中被decision tree模型错误分类的样本
    # print("删除需要修复的样本后，完整数据集D中被decision tree模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))
    #       /(len(y_train_copy_repair) + len(y_test_copy_repair)))

    # section 方案五：训练机器学习模型（随机森林模型），修复标签值

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_absolute_error

    # subsection 修复标签值
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_copy_inners, y_inners)  # 使用正常样本训练模型

    # 预测离群样本的标签
    y_repair_pred = model.predict(X_copy_repair)

    # 计算预测的准确性（可选）
    mae = mean_absolute_error(y_repair, y_repair_pred)
    print(f'Mean Absolute Error: {mae}')

    # subsection 修复特征值

    X_copy[X_copy_repair_indices] = X_copy_repair
    y[X_copy_repair_indices] = y_repair_pred
    X_train_copy = X_copy[train_indices]
    X_test_copy = X_copy[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # subsection 重新在修复后的数据上训练decision tree模型

    # # 定义一个超参数网格
    # param_grid = {
    #     'max_depth': [5, 10, 15, None],  # 树的最大深度
    #     'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
    #     'min_samples_leaf': [1, 2, 4],  # 叶节点的最小样本数
    #     'max_features': ['auto', 'sqrt', 'log2', None],  # 最大特征数
    #     'class_weight': ['balanced', None],  # 类别权重
    #     'criterion': ['gini', 'entropy']  # 划分标准
    # }
    #
    # # 初始化决策树模型
    # dt_repair = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    #
    # # 初始化GridSearchCV进行超参数调优，使用验证集进行交叉验证
    # grid_search = GridSearchCV(estimator=dt_repair, param_grid=param_grid,
    #                            cv=2,  # 3折交叉验证
    #                            scoring='accuracy',  # 使用准确度作为评分指标
    #                            n_jobs=-1,  # 使用所有CPU核进行并行计算
    #                            verbose=2)  # 打印详细的进度信息
    #
    # # 在验证集上进行网格搜索
    # grid_search.fit(X_val, y_val)
    #
    # # 输出最佳参数和最佳交叉验证得分
    # print("最佳超参数组合：", grid_search.best_params_)
    # print("最佳交叉验证准确度：", grid_search.best_score_)
    #
    # # 使用最佳超参数训练模型
    # dt_repair = grid_search.best_estimator_
    # dt_repair.fit(X_train_copy, y_train)
    dt_repair = dt_model
    y_train_pred = dt_repair.predict(X_train_copy)
    y_test_pred = dt_repair.predict(X_test_copy)

    # # section 方案六：训练机器学习模型(随机森林模型)，修复特征值（修复时间很久，慎用）
    # #  依次将有影响力的特征作为要修复的标签（连续特征对应回归模型，分类特征对应分类模型），使用其他特征参与训练
    #
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.metrics import mean_absolute_error
    #
    # # subsection 修复特征值
    #
    # for i in top_k_indices:
    #     y_train_inf = X_copy_inners[:, i]
    #     columns_to_keep = np.delete(range(X_copy_inners.shape[1]), i)
    #     X_train_remain = X_copy_inners[:, columns_to_keep]
    #     if i in categorical_features:
    #         model = RandomForestClassifier(n_estimators=100, random_state=42)
    #     else:
    #         model = RandomForestRegressor(n_estimators=100, random_state=42)
    #     model.fit(X_train_remain, y_train_inf)  # 使用正常样本训练模型
    #     X_test_repair = X_copy_repair[:, columns_to_keep]
    #     y_test_pred = model.predict(X_test_repair)
    #     X_copy_repair[:, i] = y_test_pred
    #
    # X_copy[X_copy_repair_indices] = X_copy_repair
    # X_train_copy = X_copy[train_indices]
    # X_test_copy = X_copy[test_indices]
    # y_train = y[train_indices]
    # y_test = y[test_indices]
    #
    # # subsection 重新在修复后的数据上训练decision tree模型
    #
    # dt_repari = tree.DecisionTreeClassifier(class_weight='balanced')
    # dt_repari.fit(X_train_copy, y_train)
    # y_train_pred = dt_repari.predict(X_train_copy)
    # y_test_pred = dt_repari.predict(X_test_copy)
    #
    # print("*" * 100)
    # # 训练样本中被decision tree模型错误分类的样本
    # wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    # print("加噪标签修复后，训练样本中被decision tree模型错误分类的样本占总训练样本的比例：", len(wrong_classified_train_indices)/len(y_train))
    #
    # # 测试样本中被decision tree模型错误分类的样本
    # wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    # print("加噪标签修复后，测试样本中被decision tree模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices)/len(y_test))
    #
    # # 整体数据集D中被decision tree模型错误分类的样本
    # print("加噪标签修复后，完整数据集D中被decision tree模型错误分类的样本占总完整数据的比例：",
    #       (len(wrong_classified_train_indices) + len(wrong_classified_test_indices))/(len(y_train) + len(y_test)))

    # section 修复后实验指标测定

    """Accuracy指标"""
    print("*" * 100)
    print("分类器在修复后的加噪测试集中的分类准确度：" + str(accuracy_score(y_test, y_test_pred)))

    """Precision/Recall/F1指标"""

    # average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
    # average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
    # average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
    # average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。

    print("分类器在修复后的加噪测试集中的分类F1分数：" + str(f1_score(y_test, y_test_pred, average='weighted')))
    print("分类器在修复后的加噪测试集中的分类精确度：" + str(precision_score(y_test, y_test_pred, average='weighted')))
    print("分类器在修复后的加噪测试集中的分类召回率：" + str(recall_score(y_test, y_test_pred, average='weighted')))