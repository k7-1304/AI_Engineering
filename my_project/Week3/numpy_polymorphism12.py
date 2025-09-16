import numpy as np

class ManualTester:
    def analyse(self, data):
        print("Manual tester first 5 execution times:", data[:5])

class AutomationTester:
    def analyse(self, data):
        print("Automation tester fastest execution times:", data.min())

class PerformanceTester:
  
   def analyse(self, data):
    print("Performance tester 95% execution time:", np.percentile(data,95))

def show_analysis(tester, data):
    tester.analyse(data)

if __name__ == "__main__":
    execution_times = np.array([12.5, 15.3, 9.8, 20.1, 11.4, 2.5, 3.1, 4.0, 1.8, 5.5, 6.7, 2.9, 4.5, 3.8, 7.2])
    
    manual_tester = ManualTester()
    automation_tester = AutomationTester()
    performance_tester = PerformanceTester()
    
    show_analysis(manual_tester, execution_times)
    show_analysis(automation_tester, execution_times)
    show_analysis(performance_tester, execution_times)