import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import zscore
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.ensemble import IsolationForest
# 屏蔽 sklearn 的所有 warning（包括特征名不匹配的）
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from deepod.models import DeepSAD, RoSAS, PReNet
from deepod.models import REPEN, SLAD, ICL, NeuTraL
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# === 6. 分别分析每个离群样本 ===
def is_outlier_zscore(val, col, threshold=1):
    z = (val - col.mean()) / col.std()
    return abs(z) > threshold, z

# 稀疏性计算函数
def compute_sparsity(outlier_indices, X_test, X_train, model, out_clf, y_test, threshold_loss=0.5):
    total_features = len(X_test.columns)  # 总特征数
    sparse_feature_count = 0  # 满足条件的特征数

    for idx in outlier_indices:
        print("*" * 100)
        print(f"\n{'='*40}\n样本 idx = {idx}")
        u = X_test.iloc[idx]
        # print("目标样本：\n", u)

        # 提取 SHAP 值
        shap_value = shap_values[idx]
        sample_values = shap_value.values
        feature_names = X.columns

        # 1. 模型重要特征（MDetI）
        important_features_dict = {}
        # print("\n模型认为的重要特征（MDetI）:")
        for name, value in zip(feature_names, sample_values):
            if abs(value) > 0.1:  # 自定义阈值
                important_features_dict[name] = value
                # print(f"  {name}: SHAP = {value:.3f}")

        # 2. 离群值检测（仅限重要特征）
        outlier_features = {}
        # print("\nOutlier 检测（仅限重要特征）:")
        for f in important_features_dict:
            col_vals = X_train[f]
            is_out, z = is_outlier_zscore(u[f], col_vals)
            if is_out:
                outlier_features[f] = z
                # print(f"  {f}: {u[f]} (z-score = {z:.2f}) → 是离群值")
            else:
                continue
                # print(f"  {f}: {u[f]} (z-score = {z:.2f}) → 正常")

        # 3. Imbalanced 检测（仅对被判为离群值的特征）
        imbalanced_features = {}
        # print("\nImbalanced 检测（仅对被判为离群值的特征）:")

        for f in outlier_features:  # 只对被判为离群的特征检测分布变化
            S = X_train[f]  # 原始训练集中的某列
            S_plus_u = pd.concat([S, pd.Series(u[f])], ignore_index=True)  # 加入当前样本值

            # === 标准差变化
            orig_std = S.std()
            new_std = S_plus_u.std()
            delta_std = abs(new_std - orig_std) / (orig_std + 1e-8)

            # === 偏度变化（衡量分布对称性）
            orig_skew = skew(S)
            new_skew = skew(S_plus_u)
            delta_skew = abs(new_skew - orig_skew)

            # === IQR 检查（衡量分布尾部异常值）
            Q1 = S.quantile(0.25)
            Q3 = S.quantile(0.75)
            IQR = Q3 - Q1

            # === 判断是否不平衡
            is_imbalanced = False

            if delta_std > 0.02:
                # print(f"  {f}：标准差变化 Δ = {delta_std:.2%}")
                is_imbalanced = True
            elif delta_skew > 0.1:
                # print(f"  {f}：偏度变化 Δ = {delta_skew:.3f}")
                is_imbalanced = True
            elif u[f] < Q1 - 1.5 * IQR or u[f] > Q3 + 1.5 * IQR:
                # print(f"  {f}：值 {u[f]} 超出 IQR 范围 ({Q1:.2f}, {Q3:.2f})")
                is_imbalanced = True

            # === 输出不平衡判断
            if is_imbalanced:
                # print(f"  → {f} 经 Imbalanced 检测为不平衡属性")
                imbalanced_features[f] = True

        # 4. Loss(u) 谓词检测
        high_loss_features = []
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([u])[0]
            true_label = y_test.iloc[idx]
            try:
                loss_val = log_loss([true_label], [proba], labels=[0, 1])
            except:
                loss_val = float("nan")

            # print(f"  🔍 log loss = {loss_val:.4f}")
            if loss_val > threshold_loss:
                # print("  → loss(u) = True，高损失样本")
                high_loss_features.append(u.name)

        # 5. 统计符合条件的特征数量
        # 对离群、不平衡和高损失的特征进行统计
        all_affected_features = set(outlier_features.keys()).union(set(imbalanced_features.keys())).union(set(high_loss_features))
        sparse_feature_count += len(all_affected_features)

    # 6. 计算稀疏性
    sparsity_score = sparse_feature_count / total_features
    # print(f"\n稀疏性评分：{sparsity_score:.4f}")
    return sparsity_score


def compute_fidelity(outlier_indices, X_test, X_train, model, out_clf, y_test, threshold_loss=0.5):
    """
    计算忠实度（Fidelity）评分。

    outlier_indices: 离群值样本的索引
    X_test: 测试集特征
    X_train: 训练集特征
    model: 主要的分类模型
    out_clf: 辅助模型（用于离群值检测）
    y_test: 测试集标签
    threshold_loss: 高损失值的阈值，默认0.5
    """

    total_outliers = len(outlier_indices)
    satisfied_count = 0  # 满足条件的离群样本数

    for idx in outlier_indices:
        # print("*" * 100)
        # print(f"\n{'=' * 40}\n样本 idx = {idx}")
        u = X_test.iloc[idx]
        # print("目标样本：\n", u)

        # 提取 SHAP 值
        shap_value = shap_values[idx]
        sample_values = shap_value.values
        feature_names = X.columns

        # 模型重要特征（MDetI）
        important_features_dict = {}
        for name, value in zip(feature_names, sample_values):
            if abs(value) > 0.1:  # 自定义阈值
                important_features_dict[name] = value

        # 离群值检测（仅限重要特征）
        outlier_features = {}
        is_outlier_feature = False
        for f in important_features_dict:
            col_vals = X_train[f]
            is_out, z = is_outlier_zscore(u[f], col_vals)
            if is_out:
                outlier_features[f] = z
                is_outlier_feature = True

        # Imbalanced 检测（仅对被判为离群值的特征）
        is_imbalanced = False
        for f in outlier_features:  # 只对被判为离群的特征检测分布变化
            S = X_train[f]  # 原始训练集中的某列
            S_plus_u = pd.concat([S, pd.Series(u[f])], ignore_index=True)  # 加入当前样本值

            # 标准差变化
            orig_std = S.std()
            new_std = S_plus_u.std()
            delta_std = abs(new_std - orig_std) / (orig_std + 1e-8)

            # 偏度变化
            orig_skew = skew(S)
            new_skew = skew(S_plus_u)
            delta_skew = abs(new_skew - orig_skew)

            # IQR 检查
            Q1 = S.quantile(0.25)
            Q3 = S.quantile(0.75)
            IQR = Q3 - Q1

            if delta_std > 0.02 or delta_skew > 0.1 or u[f] < Q1 - 1.5 * IQR or u[f] > Q3 + 1.5 * IQR:
                is_imbalanced = True

        # log_loss 检查
        proba = model.predict_proba([u])[0]
        true_label = y_test.iloc[idx]
        try:
            loss_val = log_loss([true_label], [proba], labels=[0, 1])
        except:
            loss_val = float("nan")

        # 满足 loss 阈值
        is_high_loss = loss_val > threshold_loss

        # 统计满足条件的样本
        if is_outlier_feature or is_imbalanced or is_high_loss:
            satisfied_count += 1

        # 输出每个样本的详细信息
        # print(f"  🔍 log loss = {loss_val:.4f}")
        # print(f"  → 离群值特征: {is_outlier_feature}")
        # print(f"  → 不平衡特征: {is_imbalanced}")
        # print(f"  → 高损失样本: {is_high_loss}")

    # 计算忠实度
    fidelity_score = satisfied_count / total_outliers if total_outliers > 0 else 0.0
    return fidelity_score

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
model.fit(X_train, y_train)

# === 4. 预测 & 选取前10个被识别为离群值的样本 ===
y_pred = model.predict(X_test)
outlier_indices = np.where(y_pred == 1)[0][:10]  # 前10个预测为离群值的样本

# === 5. 模型 SHAP 解释器初始化 ===
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 辅助检测模型（如 Isolation Forest）
mo_model = IsolationForest(random_state=42)
mo_model.fit(X_train)

# choice NeuTraL异常检测器
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
out_clf = NeuTraL(epochs=1, device=device)
# 转换为 float32 的 numpy 数组
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
out_clf.fit(X_tensor)

for idx in outlier_indices:
    print("*"*100)
    print(f"\n{'='*40}\n样本 idx = {idx}")
    u = X_test.iloc[idx]
    print("目标样本：\n", u)

    # 提取 SHAP 值
    shap_value = shap_values[idx]
    sample_values = shap_value.values
    feature_names = X.columns

    # 模型重要特征（MDetI）
    important_features_dict = {}
    print("\n模型认为的重要特征（MDetI）:")
    for name, value in zip(feature_names, sample_values):
        if abs(value) > 0.1:  # 自定义阈值
            important_features_dict[name] = value
            print(f"  {name}: SHAP = {value:.3f}")

    # 离群值检测（仅限重要特征）
    outlier_features = {}
    print("\nOutlier 检测（仅限重要特征）:")
    for f in important_features_dict:
        col_vals = X_train[f]
        is_out, z = is_outlier_zscore(u[f], col_vals)
        if is_out:
            outlier_features[f] = z
            print(f"  {f}: {u[f]} (z-score = {z:.2f}) → 是离群值")
        else:
            print(f"  {f}: {u[f]} (z-score = {z:.2f}) → 正常")

    print("\nImbalanced 检测（仅对被判为离群值的特征）:")

    for f in outlier_features:  # 只对被判为离群的特征检测分布变化
        S = X_train[f]  # 原始训练集中的某列
        S_plus_u = pd.concat([S, pd.Series(u[f])], ignore_index=True)  # 加入当前样本值

        # === 标准差变化
        orig_std = S.std()
        new_std = S_plus_u.std()
        delta_std = abs(new_std - orig_std) / (orig_std + 1e-8)

        # === 偏度变化（衡量分布对称性）
        orig_skew = skew(S)
        new_skew = skew(S_plus_u)
        delta_skew = abs(new_skew - orig_skew)

        # === IQR 检查（衡量分布尾部异常值）
        Q1 = S.quantile(0.25)
        Q3 = S.quantile(0.75)
        IQR = Q3 - Q1

        # === 判断是否不平衡
        is_imbalanced = False

        if delta_std > 0.02:
            print(f"  {f}：标准差变化 Δ = {delta_std:.2%}")
            is_imbalanced = True
        elif delta_skew > 0.1:
            print(f"  {f}：偏度变化 Δ = {delta_skew:.3f}")
            is_imbalanced = True
        elif u[f] < Q1 - 1.5 * IQR or u[f] > Q3 + 1.5 * IQR:
            print(f"  {f}：值 {u[f]} 超出 IQR 范围 ({Q1:.2f}, {Q3:.2f})")
            is_imbalanced = True

        # === 输出不平衡判断
        if is_imbalanced:
            print(f"  → {f} 经 Imbalanced 检测为不平衡属性")

    u = X_test.iloc[idx]
    # === Loss(u) 谓词 ===
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([u])[0]
        true_label = y_test.iloc[idx]
        try:
            loss_val = log_loss([true_label], [proba], labels=[0, 1])
        except:
            loss_val = float("nan")

        print(f"  🔍 log loss = {loss_val:.4f}")
        if loss_val > 0.5:
            print("  → loss(u) = True，高损失样本")
        else:
            print("  → loss(u) = False，损失正常")

    print("\n🧪 Mo 谓词检测（使用辅助模型 out_clf，Tensor 输入）:")

    try:
        # 将 u 转为 float32 Tensor，并添加 batch 维度（即 shape [1, d]）
        mo_input = torch.tensor([u.values], dtype=torch.float32).to(out_clf.device)

        # === 调用已有的模型推理接口 ===
        mo_label = out_clf.predict(mo_input)[0]  # 通常 1 表示正常，-1 表示离群

        # === 可选输出：confidence 分数（如果模型有对应方法）
        if hasattr(out_clf, 'decision_function'):
            mo_score = out_clf.decision_function(mo_input)[0]
            print(f"  → 离群判断: {mo_label}（score = {mo_score:.4f}）")
        elif hasattr(out_clf, 'predict_proba'):
            proba = out_clf.predict_proba(mo_input)[0]
            print(f"  → 离群判断: {mo_label}（proba = {proba}）")
        else:
            print(f"  → 离群判断: {mo_label}")

        # === 判定 Mo(u)
        if mo_label == -1:
            print("  → Mo(u) = True，被 out_clf 判定为离群值")
        else:
            print("  → Mo(u) = False，out_clf 判定为正常")

    except Exception as e:
        print("  → Mo 检测失败：", e)

    # === Mo(u) 谓词 ===
    print("  🧪 Mo 谓词检测（使用辅助模型）:")
    mo_pred = mo_model.predict([u])
    if mo_pred[0] == -1:
        print("  → Mo(u) = True，被辅助模型判定为离群值")
    else:
        print("  → Mo(u) = False，辅助模型认为是正常样本")

"""
计算稀疏性：
稀疏性（Sparsity） 衡量了LIME解释中使用的特征数量相对于模型特征总数的比例。通过计算这一比例，您可以量化解释的简洁程度，并进一步研究用户对于不同稀疏性水平的接受度。
 """
# 遍历 outlier_indices 中的每个样本，计算稀疏性
sparsity_scores = []
for idx in outlier_indices:
    idx_array = np.array([idx])  # 将 idx 转换为 numpy 数组
    sparse_score = compute_sparsity(idx_array, X_test, X_train, model, out_clf, y_test, threshold_loss=0.5)
    sparsity_scores.append(sparse_score)

# 输出所有样本的稀疏性分数
print("这些样本的稀疏性分数为：")
for idx, score in zip(outlier_indices, sparsity_scores):
    print(f"样本 idx = {idx}, 稀疏性分数 = {score:.4f}")

# 计算均值
mean_sparsity = np.mean(sparsity_scores)

# 打印汇报结果
print(f"平均稀疏性评分为：{mean_sparsity:.4f}")

"""
计算忠实度：
 """
# 使用该函数计算忠实度评分
print("="*30)
fidelity_score = compute_fidelity(outlier_indices, X_test, X_train, model, out_clf, y_test, threshold_loss=0.5)
print("这些离群样本的忠实度评分为：", fidelity_score)

"""
计算忠实度：
 """
# # 使用该函数计算忠实度评分
# print("="*30)
# fidelity_scores = []
# for idx in outlier_indices:
#     idx_array = np.array([idx])  # 将 idx 转换为 numpy 数组
#     fidelity_score = compute_fidelity(idx_array, X_test, X_train, model, out_clf, y_test, threshold_loss=0.5)
#     fidelity_scores.append(fidelity_score)
# # 输出所有样本的稀疏性分数
# print("这些样本的忠实度分数为：")
# for idx, score in zip(outlier_indices, fidelity_scores):
#     print(f"样本 idx = {idx}, 稀疏性分数 = {score:.4f}")