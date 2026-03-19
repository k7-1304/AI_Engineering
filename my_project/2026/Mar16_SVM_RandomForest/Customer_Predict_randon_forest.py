import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score

df = pd.read_csv('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/Assignment/2026/Mar16_SVM_RandomForest/customer_purchase_data1.csv')
print(df.head())

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

#model evaluation
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

print("Decision Tree Precision:", precision_score(y_test, dt_pred))
print("Random Forest Precision:", precision_score(y_test, rf_pred))

print("Decision Tree Recall:", recall_score(y_test, dt_pred))
print("Random Forest Recall:", recall_score(y_test, rf_pred))

importance = pd.Series(rf.feature_importances_, index=X.columns)
print("Feature Importances:\n", importance.sort_values(ascending=False))

#prediction

age = int(input("Enter Age: "))
salary = float(input("Enter Estimated Salary: "))

input_data = pd.DataFrame([[age, salary]], columns=['Age', 'EstimatedSalary'])

dt_prediction = dt.predict(input_data)
rf_prediction = rf.predict(input_data)

print("Decision Tree Prediction:", dt_prediction[0])
print("Random Forest Prediction:", rf_prediction[0])

if rf_prediction[0] == 1:
    print("Prediction: Customer will PURCHASE the product.")
else:
    print("Prediction: Customer will NOT purchase the product.")
