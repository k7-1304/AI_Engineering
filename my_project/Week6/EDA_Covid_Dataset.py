from Class_Covid_case import CovidAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Create a class CovidEDA to perform the below operations. 


class CovidEDA(CovidAnalysis):
    def __init__(self,file_path):
        super().__init__(file_path)

# Load the dataset using Pandas.
# Keep only the columns Confirmed and New cases for analysis.

    def load_and_filter_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df = self.df[['Confirmed','New cases']]
        print("Filtered Data:\n", self.df.head())

#Compute Statistical Measures Calculate and print: Mean, Median, Variance, Standard Deviation, Correlation Matrix (between Confirmed and New cases)

    def compute_statistical_measures(self):
         
         mean_confirmed = self.df['Confirmed'].mean()
         median_confirmed = self.df['Confirmed'].median()
         variance_confirmed = self.df['Confirmed'].var()
         std_confirmed = self.df['Confirmed'].std()
    
         mean_newcases = self.df['New cases'].mean()
         median_newcases=self.df['New cases'].median()
         variance_newcases = self.df['New cases'].var()
         std_newcases = self.df['New cases'].std()
         print(f"Confirmed Cases - Mean: {mean_confirmed}, Median: {median_confirmed}, Variance: {variance_confirmed}, Standard Deviation: {std_confirmed}")
         print(f"New Cases - Mean: {mean_newcases}, Median: {median_newcases}, Variance: {variance_newcases}, Standard Deviation: {std_newcases}")  

         #correlation_matrix = self.df.corr()
         correlation_matrix = self.df['Confirmed'].corr(self.df['New cases'])
    
         print("Correlation Matrix:\n", correlation_matrix)

# Outlier Detection using IQR Technique Identify outliers in both Confirmed and New cases. Remove the outliers and store the cleaned data in a new DataFrame. Display the cleaned dataset.

    def detect_remove_outliers(self):
        q1_cnfr = self.df['Confirmed'].quantile(0.25)
        q3_cnfr = self.df['Confirmed'].quantile(0.75)
        iqr_cnfr = q3_cnfr - q1_cnfr
        lower_bound_cnfr = q1_cnfr - 1.5 * iqr_cnfr
        upper_bound_cnfr = q3_cnfr + 1.5 * iqr_cnfr

        q1_new = self.df['New cases'].quantile(0.25)
        q3_new = self.df['New cases'].quantile(0.75)
        iqr_new = q3_new - q1_new
        lower_bound_new = q1_new - 1.5 * iqr_new
        upper_bound_new = q3_new + 1.5 * iqr_new

        outliers_cnfr = self.df[(self.df['Confirmed'] < lower_bound_cnfr) | (self.df['Confirmed'] > upper_bound_cnfr)]
        outliers_new = self.df[(self.df['New cases'] < lower_bound_new)
                    | (self.df['New cases'] > upper_bound_new)]
        print("Outliers in Confirmed Cases:\n", outliers_cnfr)
        print("Outliers in New Cases:\n", outliers_new)
        self.cleaned_df = self.df[(self.df['Confirmed'] >= lower_bound_cnfr) & (self.df['Confirmed'] <= upper_bound_cnfr) &
                             ((self.df['New cases'] >= lower_bound_new) & (self.df['New cases'] <= upper_bound_new))]
        print("Cleaned Data (Outliers Removed):\n", self.cleaned_df)
        self.cleaned_df.to_csv('cleaned_covid_data.csv', index=False)

#  Normalization using Standard Scaler Apply StandardScaler from sklearn.preprocessing to normalize the Confirmed and New Cases.
# Display the scaled (normalized) output as a new DataFrame.

    def normalize_data(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.cleaned_df[['Confirmed','New cases']])
        scaled_df = pd.DataFrame(scaled_data, columns=['Confirmed', 'New cases'])
        print("Scaled Data:\n", scaled_df.head())
        scaled_df.to_csv('scaled_covid_data.csv', index=False)

# Visualization Tasks
# Plot Histograms for Confirmed and New cases (before and after normalization) using Seaborn, to visualize the bell curve.


    def plot_histogram(self):

        plt.hist(self.df['Confirmed'], bins =15, color = 'blue', alpha = 0.7, label = 'Confirmed Cases - Original')
        plt.hist(self.df['New cases'], bins = 15, color = 'orange', alpha = 0.7, label= 'New Cases - Original')
        plt.title('Histogram of Confirmed and New Cases - Original')
        plt.tight_layout()
        plt.show()

        plt.hist(self.cleaned_df['Confirmed'], bins =15, color = 'green', alpha = 0.7, label = 'Confirmed Cases - Cleaned')
        plt.hist(self.cleaned_df['New cases'], bins = 15, color = 'red', alpha = 0.7, label= 'New Cases - Cleaned')
        plt.title('Histogram of Confirmed and New Cases - Cleaned')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Plot a Heatmap between Confirmed and New cases to display their correlation visually.

        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Heatmap of Correlation between Confirmed and New Cases')
        plt.show()

if __name__ == "__main__":
    Covid_EDA = CovidEDA('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/country_wise_latest.csv')
    Covid_EDA.load_and_filter_data()
    Covid_EDA.compute_statistical_measures()
    Covid_EDA.detect_remove_outliers()
    Covid_EDA.normalize_data()
    Covid_EDA.plot_histogram()