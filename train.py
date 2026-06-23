from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset directly (built into scikit-learn)
housing = fetch_california_housing()

# Convert to a DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

print(df.head())
print("\nShape:", df.shape)
print("\nColumn names:", df.columns.tolist())






import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Basic info
print("\n--- Dataset Info ---")
print(df.describe())

# Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Chart 1: Price Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Price'], bins=50, color='blue')
plt.title('House Price Distribution')
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.show()
print("Chart 1 saved!")

# Chart 2: Correlation Heatmap (which features affect price most?)
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

# Step 1: Separate features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Step 2: Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 3: Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Step 4: Predict on test data
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
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