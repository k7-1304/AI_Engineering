import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score)

# Load the dataset using Pandas
df = pd.read_csv('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/Assignment/2026/Week22/Social_Network_Ads.csv')

#First few rows
print("First few rows of the dataset:")
print(df.head())
#Dataset shape
print("Dataset shape (rows, columns):")
print(df.shape)
#Summary statistics
print("Summary statistics:")
print(df.describe())

#Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

label_encoder = LabelEncoder()
df['Purchased_encoded'] = label_encoder.fit_transform(df['Purchased'])

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building – Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict:
# o Class labels for test data
# o Probabilities for test data

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]



#Confusion Matrix Display the matrix and explain:
#  True Positives
#  True Negatives
#  False Positives
#  False Negatives
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Metrics (Threshold: 0.5)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Purchased', 'Purchased']))

#Visualization 
# Plot the Confusion Matrix using Matplotlib
# Clearly label:
# o Axes
# o Class names (Purchased / Not Purchased)

plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
classes = ["Not Purchased", "Purchased"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.show()



# User Input Prediction
age = int(input("Enter Age: "))
estimated_salary = float(input("Enter Estimated Salary: "))

# Create DataFrame for the new input
user_input = pd.DataFrame({
    'Age': [age],
    'EstimatedSalary': [estimated_salary]
})

# Predict purchase probability
purchase_prob = model.predict_proba(user_input)[:, 1]
final_prediction = model.predict(user_input)

print(f"Purchase Probability: {purchase_prob[0]:.2f}")
print(f"Final Prediction: {'Purchase' if final_prediction[0] == 1 else 'Not Purchase'}")


