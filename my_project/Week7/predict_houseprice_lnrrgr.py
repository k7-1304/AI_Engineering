import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

#Load the Dataset
# Load the provided dataset using Pandas.
# Retain only the columns Square Footage and Price for model building.

data = pd.read_csv('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/Assignment/Week7/house_price_regression_dataset.csv')

data = data[['Square_Footage','House_Price']]

#Display the first few rows of the dataset.
print("Filtered data:\n",data.head())

#Exploratory Data Analysis (EDA)
# Check for missing or null values and handle them appropriately.
# Visualize the relationship between Square Footage and Price using a scatter plot.

print(data.isnull())
data = data.dropna()

sns.scatterplot(x='Square_Footage', y='House_Price', data=data)
plt.title('Square Footage vs House Price')
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.show()

#Feature and Target Selection
# Assign Square Footage as the independent variable (X).
# Assign Price as the dependent variable (Y).

X = data[['Square_Footage']]
y = data['House_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Model Building 
# Create a Linear Regression model using LinearRegression from sklearn.linear_model.
# Fit the model on the training data. Display the intercept (b₀) and coefficient (b₁) of the regression line.

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Intercept (b0):", model.intercept_)
print("Coefficient (b1):", model.coef_)

# Prediction and Evaluation
# Predict the house prices for the test set.
# Calculate and print the following evaluation metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score (Coefficient of Determination)

print("MSE: ", mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2:", r2_score(y_test, y_pred))

#Visualization. Plot the regression line along with the actual data points.
# Visualize actual vs predicted prices to assess model performance.

plt.scatter(X_test, y_test, color = 'red', label =  'Actual')
plt.plot(X_test, y_pred, color = 'orange', label = 'Predicted')
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.title('House Price Prediction')
plt.legend()
plt.show()