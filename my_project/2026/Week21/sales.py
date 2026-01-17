import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/Assignment/2026/Week21/sales_data (1).csv')

plt.scatter(df['Advertising_Spend'], df['Total Amount'], color='red', label='Raw data plot')
plt.xlabel('Advertising Spend')
plt.ylabel('Total Amount')
plt.legend()
plt.show()

adv_Q1 = df['Advertising_Spend'].quantile(0.25)
adv_Q3 = df['Advertising_Spend'].quantile(0.75)

adv_IQR = adv_Q3 - adv_Q1
lower_limit = adv_Q1 - 1.5 * adv_IQR
upper_limit = adv_Q3 + 1.5 * adv_IQR
adv_outliers_removed = df[(df['Advertising_Spend'] > lower_limit) & (df['Advertising_Spend'] < upper_limit)]
print(adv_outliers_removed)

X = adv_outliers_removed[['Advertising_Spend']].to_numpy().reshape(-1, 1)
Y = adv_outliers_removed[['Total Amount']].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, Y_train)

Y_pred = model.predict(X_test_poly)
sorted_idx = X_test.flatten().argsort()

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

model2 = LinearRegression()
model2.fit(X_train, Y_train)
Y_pred_linear = model2.predict(X_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Polynomial Regression Plot
sorted_idx = X_test.flatten().argsort()
ax1.scatter(X_test, Y_test, color='red', label='Actual', alpha=0.5, marker='o')
ax1.plot(
    X_test.flatten()[sorted_idx],
    Y_pred[sorted_idx],
    color='green',
    label='Predicted',
    marker='o'
)
ax1.set_title('Polynomial Regression (Degree 3)')
ax1.set_xlabel('Advertising Spend')
ax1.set_ylabel('Total Amount')
ax1.legend()

# Linear Regression Plot
sorted_idx = X_test.flatten().argsort()
ax2.scatter(X_test, Y_test, color='red', label='Raw data plot')
ax2.plot(X_test.flatten()[sorted_idx], Y_pred_linear[sorted_idx], color='green', label='Linear Regression')
ax2.set_xlabel('Advertising Spend')
ax2.set_ylabel('Total Amount')
ax2.legend()
plt.tight_layout()
plt.show()



advertising_spend = float(input("Enter Advertising Spend: "))
ad_spend_poly = poly.transform([[advertising_spend]])
predicted_sales = model.predict(ad_spend_poly)
print(f"Predicted Total Amount for Advertising Spend {advertising_spend}: {predicted_sales[0]}")
