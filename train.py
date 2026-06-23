from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

print(df.head())
print("\nShape:", df.shape)
print("\nColumn names:", df.columns.tolist())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("\n--- Dataset Info ---")
print(df.describe())
print("\n--- Missing Values ---")
print(df.isnull().sum())

plt.figure(figsize=(8, 4))
sns.histplot(df['Price'], bins=50, color='blue')
plt.title('House Price Distribution')
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.show()
print("Chart 1 saved!")

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Chart 2 saved!")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X = df.drop('Price', axis=1)
y = df['Price']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully!")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"MAE  (Mean Absolute Error): {mae:.4f}")
print(f"RMSE (Root Mean Sq Error) : {rmse:.4f}")
print(f"R²   (Accuracy Score)     : {r2:.4f}")   



import pickle
import gzip

with gzip.open('house_price_model.pkl.gz', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully (compressed)!")