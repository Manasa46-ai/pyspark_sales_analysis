# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load dataset
data = pd.read_csv("house_prices.csv")
print("Dataset preview:")
print(data.head())

# Step 3: Features (X) and Target (y)
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Step 4: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Step 8: Visualize Actual vs Predicted
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Step 9: Predict new house price
new_house = [[3000, 4, 3]]  # 3000 sqft, 4 bedrooms, 3 bathrooms
prediction = model.predict(new_house)
print("\nPredicted Price for 3000 sqft, 4BHK, 3 Bath:", prediction[0])
