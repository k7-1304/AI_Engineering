import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

df = pd.read_csv('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/Assignment/2026/Mar16_SVM_RandomForest/customer_purchase_data1.csv')
print(df.head())

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("SVM Precision:", precision_score(y_test, y_pred))
print("SVM Recall:", recall_score(y_test, y_pred))
