import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns

class StudentPerformanceModel:

    def __init__(self, data_path):
        self.data_path = data_path
        self.model = LinearRegression()

#Data Preprocessing
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df.dropna(inplace=True)
        print("Data Loaded Successfully. Sample Data:\n", self.df.head())

        self.X = self.df[['Hours Studied', 'Sleep Hours', 'Previous Scores']]
        self.y = self.df['Performance Index']

#Train-Test Split
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        print("Data Split into Training and Testing Sets.")
        print(f"Training Samples: {len(self.X_train)}, Testing Samples: {len(self.X_test)}")
#Model Training
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model Trained Successfully.")
        print(f"Intercept (β0): {self.model.intercept_:.2f}")
        print(f"Coefficients (β1, β2, β3): {self.model.coef_}")
        print(f"Model Equation: Performance Index = {self.model.intercept_:.2f} + "
              f"{self.model.coef_[0]:.2f}*Hours Studied + "
              f"{self.model.coef_[1]:.2f}*Sleep Hours + "
              f"{self.model.coef_[2]:.2f}*Previous Scores")
       
#Model Evaluation

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        print("\n--- Model Performance Evaluation ---")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")

#3D Visualization
    def visualize_results(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.X_test['Hours Studied'], self.X_test['Sleep Hours'], self.y_test,
                   color='blue', s=100, label='Actual Data Points')

        x1_surf = np.linspace(self.X['Hours Studied'].min(), self.X['Hours Studied'].max(), 10)
        x2_surf = np.linspace(self.X['Sleep Hours'].min(), self.X['Sleep Hours'].max(), 10)
        x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)

        y_plane = self.model.predict(pd.DataFrame({
            'Hours Studied': x1_surf.ravel(),
            'Sleep Hours': x2_surf.ravel(),
            'Previous Scores': np.full(x1_surf.ravel().shape, self.X['Previous Scores'].mean())
        }))
        y_plane = y_plane.reshape(x1_surf.shape)

        ax.plot_surface(x1_surf, x2_surf, y_plane, color='orange', alpha=0.5, label='Predicted Plane')
        ax.set_xlabel('Hours Studied')
        ax.set_ylabel('Sleep Hours')
        ax.set_zlabel('Performance Index')
        ax.set_title('Student Performance Prediction')
        ax.legend()
        plt.show()

#Predict Performance for New Student
    def predict_performance(self):
        print("\n--- Predict Performance for New Student ---")
        hours_studied = float(input("Enter Hours Studied: "))
        sleep_hours = float(input("Enter Sleep Hours: "))
        previous_scores = float(input("Enter Previous Scores: "))
        new_data = pd.DataFrame({'Hours Studied': [hours_studied],
                                 'Sleep Hours': [sleep_hours],
                                 'Previous Scores': [previous_scores]})
        predicted_performance = self.model.predict(new_data)
        print(f"Predicted Performance Index: {predicted_performance[0]:.2f}")

if __name__ == "__main__":
    data_path = '/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/Assignment/Week9/Student_Performance.csv' 
    spm = StudentPerformanceModel(data_path)
    spm.load_data()
    spm.split_data()
    spm.train_model()
    spm.evaluate_model()
    spm.visualize_results()
    spm.predict_performance()
