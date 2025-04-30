from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = AutoTabPFNClassifier(max_time=120, device="auto") # 120 seconds tuning time
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)