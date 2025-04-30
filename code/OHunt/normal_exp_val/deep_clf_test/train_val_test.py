"""
验证集的主要用途是：
    调整超参数：使用验证集来选择最佳的超参数。
    选择清洗策略：尝试不同的数据预处理或特征工程技术，并通过验证集来评估它们的效果。
    调整模型架构：可以通过验证集来调整不同模型的选择和配置。

GridSearchCV 通过 交叉验证 来选择超参数。
在每次交叉验证时，数据集会被划分成多个部分（比如 3 折交叉验证，数据集会被分成 3 个子集），每个子集轮流作为验证集，其余部分作为训练集。
GridSearchCV 会在训练集上训练模型，并在验证集上评估性能。最终，它会根据验证集上的表现选择最佳的超参数组合。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 加载数据
X, y = load_breast_cancer(return_X_y=True)

# 划分数据集：70%训练集，20%验证集，10%测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)  # 0.33 * 0.3 = 0.1

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 定义需要调整的超参数
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}  # 逻辑回归的正则化参数C的不同值

# 定义逻辑回归模型
log_reg = LogisticRegression(max_iter=10000)

# 使用 GridSearchCV 进行超参数搜索，交叉验证分 3 折
# 超参数选择：
#     在 param_grid 中定义了我们要搜索的超参数空间。这里，我们正在调整 C（正则化参数）来控制逻辑回归模型的正则化强度。
#     GridSearchCV 会在训练集上使用 3 折交叉验证（cv=3），并在每个折上使用验证集评估每个超参数设置的性能。
# 例如：
#     在第一个折中，训练数据集的 2/3 用于训练模型，剩下的 1/3 用于验证模型性能。这个过程会对每个超参数组合进行重复，直到所有的超参数都被评估过。
#     在每一轮训练中，GridSearchCV 会基于训练数据（训练集的一部分）训练模型，并使用 验证集（交叉验证过程中的验证集）评估模型的表现，最终选出最佳的超参数。
grid_search = GridSearchCV(log_reg, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# 在训练集上训练并使用交叉验证进行超参数选择
grid_search.fit(X_train_scaled, y_train)

# 打印最优的超参数
print(f"Best hyperparameters: {grid_search.best_params_}")

# 使用最优的超参数重新训练模型
best_model = grid_search.best_estimator_

# 在验证集上评估模型
y_val_pred = best_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 在测试集上进行最终评估
y_test_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

