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

# === 1. 加载数据 ===
file_path = "../../datasets/real_outlier/Cardiotocography.csv"
# file_path = "../../datasets/real_outlier/annthyroid.csv"
# file_path = "../../datasets/real_outlier/optdigits.csv"
# file_path = "../../datasets/real_outlier/PageBlocks.csv"
# file_path = "../../datasets/real_outlier/pendigits.csv"
# file_path = "../../datasets/real_outlier/satellite.csv"
# file_path = "../../datasets/real_outlier/shuttle.csv"
# file_path = "../../datasets/real_outlier/yeast.csv"
data = pd.read_csv(file_path)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

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

# === 4. 预测 & 选取某条被识别为离群值的样本 ===
y_pred = model.predict(X_test)
outlier_idx = np.where(y_pred == 1)[0][0]  # 第一个预测为离群值的样本
u = X_test.iloc[outlier_idx]
print("目标样本：\n", u)

# === 5. MDetI(): 模型重要特征解释 ===
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 展示目标样本的重要特征
shap.plots.waterfall(shap_values[outlier_idx], max_display=10)

# 提取特征重要性
important_features = shap_values[outlier_idx].values
feature_names = X.columns

# === 5. 模型解释：找出重要特征（MDetI） ===
important_features_dict = {}  # 存储重要特征及其 SHAP 值

print("\n模型认为的重要特征（MDetI）:")
for name, value in zip(feature_names, important_features):
    if abs(value) > 0.1:  # 你设定的阈值
        important_features_dict[name] = value
        print(f"  {name}: SHAP = {value:.3f}")

# choice NeuTraL异常检测器
epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_trans = 64
random_state = 42
out_clf = NeuTraL(epochs=1, device=device)
# 转换为 float32 的 numpy 数组
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
out_clf.fit(X_tensor)

# === 6. 只在重要特征中做 Outlier 检测（Z-score 方法） ===
def is_outlier_zscore(val, col, threshold=1):
    z = (val - col.mean()) / col.std()
    return abs(z) > threshold, z

# === 6. 离群值检测：只检测重要特征 ===
outlier_features = {}  # 存储被判定为离群值的特征及其 z-score
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

# 判断分类模型
if hasattr(model, "predict_proba"):
    # 二分类情况下，计算 log loss
    proba = model.predict_proba([u])[0]
    true_label = y_test.iloc[outlier_idx]
    loss_val = log_loss([true_label], [proba], labels=[0, 1])

    print(f"  样本的 log loss = {loss_val:.4f}")
    if loss_val > 0.5:
        print("  → loss(u) = True，高损失样本")
    else:
        print("  → loss(u) = False，损失正常")

print("\n🧪 Mo 谓词检测（使用辅助模型）:")

# 使用 Isolation Forest 进行辅助检测
mo_model = IsolationForest(random_state=42)
mo_model.fit(X_train)

mo_pred = mo_model.predict([u])  # 注意：输出为 1（正常）或 -1（离群）
if mo_pred[0] == -1:
    print("  → Mo(u) = True，被辅助模型判定为离群值")
else:
    print("  → Mo(u) = False，辅助模型认为是正常样本")

try:
    # 将 u 转为 float32 Tensor，并添加 batch 维度（即 shape [1, d]）
    mo_input = torch.tensor([u.values], dtype=torch.float32).to(out_clf.device)

    # === 调用已有的模型推理接口 ===
    mo_label = out_clf.predict(mo_input)[0]  # 通常 1 表示正常，-1 表示离群

    if mo_pred[0] == -1:
        print("  → Mo(u) = True，被deepod模型判定为离群值")
    else:
        print("  → Mo(u) = False，deepod模型认为是正常样本")

except Exception as e:
    print("  → Mo 检测失败：", e)

