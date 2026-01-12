import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/Assignment/2026/Week21/sales_data (1).csv')

print(df.head())
print(df.describe())

plt.scatter(df['Advertising_Spend'], df['Total Amount'], color='red')
plt.xlabel('Advertising Spend')
plt.ylabel('Total Amount')
plt.title('Total Amount vs Advertising Spend')
plt.show()

# Step 2: Baseline Model – Simple Linear Regression

X = df[['Advertising_Spend']]
y = df['Total Amount']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# also predict on the full X so we can compare models on the same domain (used for final plotting)
y_pred_full = model.predict(X)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Simple Linear Regression")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"RMS Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color = 'red', label =  'Actual')
plt.plot(X_test, y_pred, color = 'orange', label = 'Predicted')
plt.xlabel('Advertising Spend')
plt.ylabel('Total Amount')
plt.title('Total Amount vs Advertising Spend (Test Set)')
plt.legend()
plt.show()

# Step 3: Polynomial Feature Transformation

poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)

poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)

#Step 4: Polynomial Regression Model Training
model2 = LinearRegression()
model2.fit(X_poly2,y)

y_pred2 = model2.predict(X_poly2)

model3 = LinearRegression()
model3.fit(X_poly3,y)

y_pred3 = model3.predict(X_poly3)

#Step 5: Model Evaluation and Comparison

mse_poly2 = mean_squared_error(y, y_pred2)
rmse_poly2 = np.sqrt(mse_poly2)
r2_poly2 = r2_score(y, y_pred2)
mse_poly3 = mean_squared_error(y, y_pred3)
rmse_poly3 = np.sqrt(mse_poly3)
r2_poly3 = r2_score(y, y_pred3)

comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Polynomial Regression (Degree 2)', 'Polynomial Regression (Degree 3)'],
    'MSE': [mse, mse_poly2, mse_poly3],
    'R^2 Score': [r2, r2_poly2, r2_poly3]
})

print(comparison_df)

# Step 6: Visualization

sorted_idx = np.argsort((X.values).flatten())
X_sorted = X.values[sorted_idx]

plt.figure(figsize=(10,6))
plt.scatter(X_sorted, y_pred_full[sorted_idx], color='red', label='Linear Regression')
plt.plot(X_sorted, y_pred3[sorted_idx], color='orange', label='Polynomial Regression (Degree 3)')
plt.xlabel('Advertising Spend')
plt.ylabel('Total Amount')
plt.title('Linear vs Polynomial Regression (Degree 3)')
plt.legend()
plt.show()

#Step 7: User Input Prediction

advertising_spend = float(input("Enter Advertising Spend: "))
ad_spend_poly = poly3.transform([[advertising_spend]])
predicted_sales = model3.predict(ad_spend_poly)
print(f"Predicted Total Amount for Advertising Spend {advertising_spend}: {predicted_sales[0]}")

#Step 8: Model Interpretation (Important) Answer the following questions in comments or markdown:
# 1. Why does Polynomial Regression perform better or worse than Linear Regression?
# - Polynomial Regression can capture non-linear relationships between features and the target variable, 
# which may lead to better performance on complex datasets.
# However, it can also overfit the training data, especially with higher-degree polynomials, 
# resulting in worse performance on unseen data.

# 2. What risks are associated with choosing a higher polynomial degree?
# - Higher polynomial degrees can lead to overfitting, 
# where the model learns the noise in the training data instead of the underlying pattern. 
# This can result in poor generalization to new data. Additionally, 
# higher-degree polynomials can be more sensitive to outliers.

# 3. In a real business scenario, which model would you choose and why?
# - The choice of model depends on the specific use case and data characteristics. 
# If the relationship between features and the target variable is known to be non-linear, 
# a Polynomial Regression model may be more appropriate. 
# However, if interpretability and simplicity are priorities, 
# a Linear Regression model might be preferred. 
# It's essential to validate the chosen model using cross-validation and performance metrics.

