import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import optuna

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)

# section 数据预处理
file_path = "../datasets/multi_class/drybean.xlsx"
data = pd.read_excel(file_path)
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

# section 数据特征缩放

# 对不同维度进行标准化
X = StandardScaler().fit_transform(X)
# 记录原始索引
original_indices = np.arange(len(X))

# 修改数据划分，将数据集划分为70%训练集，20%验证集，10%测试集
X_train, X_temp, y_train, y_temp, train_indices, temp_indices = \
    train_test_split(X, y, original_indices, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test, val_indices, test_indices = \
    train_test_split(X_temp, y_temp, temp_indices, test_size=1/3, random_state=42)  # 10%是测试集，剩余20%是验证集

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
# 从含噪数据中生成训练数据、验证数据和测试数据
X_train_copy = X_copy[train_indices]
X_val_copy = X_copy[val_indices]
X_test_copy = X_copy[test_indices]

# SECTION SVM模型的实现
# subsection 加噪数据集上训练的SVM模型在训练集和测试集中分错的样本比例

print("*" * 100)
svm_model_noise = svm.SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')
svm_model_noise.fit(X_train_copy, y_train)
train_label_pred_noise = svm_model_noise.predict(X_train_copy)
val_label_pred_noise = svm_model_noise.predict(X_val_copy)
test_label_pred_noise = svm_model_noise.predict(X_test_copy)

# 计算测试集上的准确率
accuracy_test = accuracy_score(y_test, test_label_pred_noise)
print("Test Accuracy: {:.2f}%".format(accuracy_test * 100))

# section 贝叶斯优化确定SVM的超参数
class Objective:
    def __init__(self, y_val, X_train_copy, y_train, X_val_copy):
        # Hold this implementation specific arguments as the fields of the class.
        self.y_val = y_val
        self.X_train_copy = X_train_copy
        self.y_train = y_train
        self.X_val_copy = X_val_copy

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        c = trial.suggest_float("C", 1e-2, 1e3, log=True)
        clf = svm.SVC(kernel=kernel, gamma="scale", C=c, probability=True, class_weight='balanced', random_state=0)
        clf.fit(self.X_train_copy, self.y_train)
        # 计算验证集上的准确率
        val_label_pred_noise = clf.predict(self.X_val_copy)
        accuracy_val = accuracy_score(self.y_val, val_label_pred_noise)
        return accuracy_val

file_path = "./svm_configuration.log"
lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)

storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj),
)
# Execute an optimization by using an `Objective` instance.
study = optuna.create_study(storage=storage, direction="maximize")
study.optimize(Objective(y_val, X_train_copy, y_train, X_val_copy), n_trials=3)

trial = study.best_trial
print("Objective Values: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

"""
在测试集上评估性能
"""

# 使用贝叶斯优化后的最佳超参数训练最终模型
best_kernel = trial.params['kernel']
best_C = trial.params['C']

# 使用最佳超参数重新训练SVM模型
final_model = svm.SVC(kernel=best_kernel, C=best_C, probability=True, class_weight='balanced', random_state=0)
final_model.fit(X_train_copy, y_train)

# 在测试集上进行预测
test_label_pred_final = final_model.predict(X_test_copy)
test_prob_pred_final = final_model.predict_proba(X_test_copy)[:, 1]  # 获取预测概率（对于二分类问题）

# 计算测试集上的性能评估指标
accuracy_final = accuracy_score(y_test, test_label_pred_final)
precision_final = precision_score(y_test, test_label_pred_final, average='macro', zero_division=1)  # 'macro'用于多分类
recall_final = recall_score(y_test, test_label_pred_final, average='macro', zero_division=1)
f1_final = f1_score(y_test, test_label_pred_final, average='macro', zero_division=1)

# 输出性能评估结果
print(f"Final Model Performance on Test Set:")
print(f"Accuracy: {accuracy_final:.4f}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall: {recall_final:.4f}")
print(f"F1 Score: {f1_final:.4f}")

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, test_label_pred_final)
print(f"Confusion Matrix:\n{conf_matrix}")
