# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Load dataset
df = pd.read_csv('house_prices.csv')  # Replace with your actual filename

# Step 3: Print top rows to understand data
print(df.head())

# Step 4: Drop non-numeric or irrelevant columns (adjust this as needed)
df = df.select_dtypes(include=[np.number])  # Keep numeric columns only
df = df.dropna()  # Remove missing values if any

# Step 5: Define features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']             # Output we want to predict

# Step 6: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate model
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Step 10: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()
