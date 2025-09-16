import numpy as np
class TestReport:
    def __init__(self, execution_times):
        self.execution_times = np.array(execution_times)
    def average_time(self):
        return np.mean(self.execution_times)
    def max_time(self):
        return np.max(self.execution_times)
    
class RegressionReport(TestReport):
    def __init__(self, execution_times):
        super().__init__(execution_times)
    def slow_tests(self, threshold):
        return self.execution_times[self.execution_times > threshold]
    
if __name__ == "__main__":
    report = RegressionReport([12.5, 15.3, 9.8, 20.1, 11.4, 2.5, 3.1, 4.0, 1.8, 5.5, 6.7, 2.9, 4.5, 3.8, 7.2])
    print("Average Execution Time:", report.average_time())
    print("Max Execution Time:", report.max_time())
    print("Tests slower than 12s:", report.slow_tests(12))