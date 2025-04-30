"""
tabpfn好像没有TabPFNRegressor模块
"""
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Assuming there is a TabPFNRegressor (if not, a different regressor should be used)
from tabpfn.regressor import TabPFNRegressor

# Example with RandomForestRegressor
# regressor = RandomForestRegressor()

# Load Boston Housing data
df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
X = df.data
y = df.target.astype(float)  # Ensure target is float for regression

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the regressor
regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)