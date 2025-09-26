import pandas as pd

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

class CovidAnalysis(Dataset):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.load_data()

    def summarise_count_region(self):
        region_summary = self.df.groupby('WHO Region') [["Confirmed", "Deaths", "Recovered"]].sum()
        print("Region-wise Summary:\n", region_summary)
    def filter_low_case_records(self):
        Low_case_record = self.df[self.df['Confirmed']>10]
        print("Low Case Records:\n", Low_case_record)
    def identify_highest_confirmed_region(self):
        region_summary = self.df.groupby('WHO Region') [["Confirmed", "Deaths", "Recovered"]].sum()
        highest_confirmed_region = region_summary['Confirmed'].idxmax()
        print("Region with Highest Confirmed Cases:\n", highest_confirmed_region)
    def sort_data_by_confirmed_cases(self):
        sorted_df = self.df.sort_values(by='Confirmed', ascending=False)
        sorted_df.to_csv('sorted_covid_data1.csv', index=False)
        print("Sorted Data by Confirmed Cases:\n", sorted_df)
    def top_5_countries_by_case_count(self):
        sorted_df = self.df.sort_values(by='Confirmed', ascending=False)    
        top_5 = sorted_df.head(5)
        print("Top 5 Countries by Case Count:\n", top_5)
    def lowest_death_count_region(self):
       region_summary = self.df.groupby('WHO Region') [["Confirmed", "Deaths", "Recovered"]].sum()
       Lowest_death = region_summary['Deaths'].idxmin()
       print("Region wise Lowest Death:", Lowest_death)
    def india_case_summary(self):
        India_summary = self.df[self.df['Country/Region'] == 'India']
        print("India's Case Summary:\n", India_summary)
        print(India_summary[["Confirmed", "Deaths", "Recovered"]])
    def calculate_mortality_rate_by_region(self):
        region_summary = self.df.groupby('WHO Region') [["Confirmed", "Deaths", "Recovered"]].sum()
        region_rate = region_summary["Deaths"] / region_summary ["Confirmed"] * 100
        print("Region wise Mortality Rate",region_rate)
    def compare_recovery_rates_across_regions(self):
        region_summary = self.df.groupby('WHO Region') [["Confirmed", "Deaths", "Recovered"]].sum()
        region_recovery = region_summary["Recovered"] / region_summary ["Confirmed"] * 100
        print("Region wise Recovered Rate:", region_recovery)
        
    def detect_outliers_in_case_counts(self):
       mean_val = self.df["Confirmed"].mean()
       std_val = self.df["Confirmed"].std()
       lower_bound = mean_val - 2*std_val
       upper_bound = mean_val + 2*std_val

       Outlier = self.df[(self.df["Confirmed"] < lower_bound) | (self.df["Confirmed"] > upper_bound)]
       print("Detected Outliers in Case Counts: ", Outlier[["Country/Region" , "Confirmed"]])
    def group_data_by_country_and_region(self):
        grouped = self.df.groupby(["Country/Region", "WHO Region"])
        print("Group Data by Country and Region:", grouped.head())
    def identify_zero_case_region(self):
        zero_recovery = self.df[self.df["Recovered"]==0]
        print("Regions with zero recovered cases: \n",zero_recovery[["Country/Region", "WHO Region", "Recovered"]])

if __name__ == "__main__":
    Covid_Data = CovidAnalysis('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/country_wise_latest.csv')
    
    Covid_Data.summarise_count_region()
    Covid_Data.filter_low_case_records()
    Covid_Data.identify_highest_confirmed_region()
    Covid_Data.sort_data_by_confirmed_cases()
    Covid_Data.top_5_countries_by_case_count()
    Covid_Data.lowest_death_count_region()
    Covid_Data.india_case_summary()
    Covid_Data.calculate_mortality_rate_by_region()
    Covid_Data.compare_recovery_rates_across_regions()
    Covid_Data.detect_outliers_in_case_counts()
    Covid_Data.group_data_by_country_and_region()
    Covid_Data.identify_zero_case_region()


