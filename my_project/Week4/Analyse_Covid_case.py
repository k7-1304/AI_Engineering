import numpy as np
import pandas as pd

df = pd.read_csv('/Users/kesavaramanrajendran/Documents/AI_Engineering/Python/my_project/country_wise_latest.csv')

#1. Summarize Case Counts by Region Display total confirmed, death, and recovered cases for each region.

region_summary = df.groupby('WHO Region') [["Confirmed", "Deaths", "Recovered"]].sum()
print("Region-wise Summary:\n", region_summary)


# 2. Filter Low Case Records Exclude entries where confirmed cases are < 10.

Low_case_record = df[df['Confirmed']>10]
print("Low Case Records:\n", Low_case_record)

# 3. Identify Region with Highest Confirmed Cases

highest_confirmed_region = region_summary['Confirmed'].idxmax()
print("Region with Highest Confirmed Cases:\n", highest_confirmed_region)

# 4. Sort Data by Confirmed Cases Save sorted dataset into a new CSV file.

sorted_df = df.sort_values(by='Confirmed', ascending=False)
sorted_df.to_csv('sorted_covid_data.csv', index=False)
print("Sorted Data by Confirmed Cases:\n", sorted_df)

# Top 5 Countries by Case Count

top_5 = sorted_df.head(5)
print("Top 5 Countries by Case Count:\n", top_5)

# Region with Lowest Death Count

Lowest_death = region_summary['Deaths'].idxmin()
print("Region wise Lowest Death:", Lowest_death)

#India’s Case Summary (as of April 29, 2020)

India_summary = df[df['Country/Region'] == 'India']
print("India's Case Summary:\n", India_summary)
print(India_summary[["Confirmed", "Deaths", "Recovered"]])

# Calculate Mortality Rate by Region Death-to-confirmed case ratio.

region_rate = region_summary["Deaths"] / region_summary ["Confirmed"] * 100
print("Region wise Mortality Rate",region_rate)

# Compare Recovery Rates Across Regions

region_recovery = region_summary["Recovered"] / region_summary ["Confirmed"] * 100
print("Region wise Recovered Rate:", region_recovery)

# Detect Outliers in Case Counts Use mean ± 2*std deviation.

mean_val = df["Confirmed"].mean()
std_val = df["Confirmed"].std()
lower_bound = mean_val - 2*std_val
upper_bound = mean_val + 2*std_val

Outlier = df[(df["Confirmed"] < lower_bound) | (df["Confirmed"] > upper_bound)]
print("Detected Outliers in Case Counts: ", Outlier[["Country/Region" , "Confirmed"]])

#  Group Data by Country and Region
grouped = df.groupby(["Country/Region", "WHO Region"])
print("Group Data by Country and Region:", grouped.head())

#  Identify Regions with Zero Recovered Cases

zero_recovery = df[df["Recovered"]==0]
print("Regions with zero recovered cases: \n",zero_recovery[["Country/Region", "WHO Region", "Recovered"]])
