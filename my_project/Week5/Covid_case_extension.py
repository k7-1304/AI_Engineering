from Class_Covid_case import CovidAnalysis
import matplotlib.pyplot as plt


class CovidVisualization(CovidAnalysis):
   
#Bar Chart of Top 10 Countries by Confirmed Cases

    def bar_top_10_confirmed(self):
        top10 = self.df.sort_values('Confirmed',ascending=False).head(10)
        plt.bar(top10['Country/Region'],top10['Confirmed'],color='skyblue')
        plt.title('Top 10 Countries by Confirmed Cases')
        plt.show()

#Pie Chart of Global Death Distribution by Region

    def pie_global_death_region(self):
        region_distribution = self.df.groupby('WHO Region')['Deaths'].sum()
        plt.pie(region_distribution,labels=list(region_distribution.index), autopct='%1.1f%%', shadow=True)
        plt.title('Global Death Distribution by Region')
        plt.show()

#Line Chart comparing Confirmed and Deaths for Top 5 Countries
    def line_confirmed_death_top5(self):
      

        top5 = self.df.sort_values('Confirmed',ascending=False).head(5)
        plt.plot(top5['Country/Region'],top5['Confirmed'], marker = 'o', linestyle = '--',label='Confirmed Cases')
        plt.plot(top5['Country/Region'],top5['Deaths'],marker = 'x', linestyle = '--',label='Deaths')
        plt.title('Confirmed vs Deaths for Top 5 Countries')
        plt.legend()
        plt.show()
# Scatter Plot of Confirmed Cases vs Recovered Cases
    def scatter_confirmed_recovered(self):
        plt.scatter(self.df['Confirmed'],self.df['Recovered'],color = 'g', alpha= 0.5)
        plt.xlabel('Confirmed Cases')
        plt.ylabel('Recovered Cases')
        plt.title('Confirmed Cases vs Recovered Cases')
        plt.show()
#Histogram of Death Counts across all Regions
    def histogram_death_counts(self):
        plt.hist(self.df['Deaths'],bins=15,color='g',alpha=0.7)
        plt.xlabel('Death Count')
        plt.ylabel('Frequency')
        plt.title('Histogram of Death Counts across all Regions')
        plt.show()
#Stacked Bar Chart of Confirmed, Deaths, and Recovered for 5 Selected Countries
    def stacked_bar_covid_stats(self,countries):
        selected = self.df[self.df['Country/Region'].isin(countries)][['Country/Region','Confirmed','Deaths','Recovered']]
        selected.set_index('Country/Region',inplace=True)
        selected.plot(kind='bar', stacked=True)
        plt.title('Stacked Bar Chart of Covid Stats for Selected Countries')
        plt.ylabel('Count')
        plt.show()

#Box Plot of Confirmed Cases across Regions
    def box_plot_confirmed_by_region(self):
        self.df.boxplot(column='Confirmed', by='WHO Region', grid=True)
        plt.title('Box Plot of Confirmed Cases across Regions')
        plt.suptitle('')
        plt.xlabel('WHO Region')
        plt.ylabel('Confirmed Cases')
        plt.show()     

#Trend Line: Plot Confirmed cases for India vs another chosen country (side by side comparison).

    def trend_line_india_vs_country(self,country):
        india = self.df[self.df['Country/Region']=='India']
        other = self.df[self.df['Country/Region']==country]
        plt.bar(['India'],[india['Confirmed'].sum()],label='India')
        plt.bar([country],[other['Confirmed'].sum()],label=country)
        plt.title(f'Confirmed Cases: India vs {country}')
        plt.ylabel('Confirmed Cases')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    vis = CovidVisualization('country_wise_latest.csv')
    vis.bar_top_10_confirmed()
    vis.pie_global_death_region()
    vis.line_confirmed_death_top5()
    vis.scatter_confirmed_recovered()
    vis.histogram_death_counts()
    vis.stacked_bar_covid_stats(['US','India','Brazil','Russia','France'])
    vis.box_plot_confirmed_by_region()
    vis.trend_line_india_vs_country('US')

    