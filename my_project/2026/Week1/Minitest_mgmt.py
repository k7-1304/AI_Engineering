import csv
import pandas as pd

class TestCase:
    def __init__(self,test_id,test_name,module,status="Not Executed"):
        self.test_id = test_id
        self.test_name = test_name
        self.module = module
        self.status = status

    def execute_test(self,result):
        if result in ["Passed","Failed"]:
            self.status = result
        else:
            print(f"Invalid result '{result}' for test case '{self.test_name}'.")

    def display_test_case(self):
        print(f"Test ID: {self.test_id}, Test Name: {self.test_name}, Module: {self.module}, Status: {self.status}")

    def to_csv_row(self):
        return [self.test_id, self.test_name, self.module, self.status,"N/A"]
    
class AutomatedTestCase(TestCase):
    def __init__(self,test_id,test_name,module,automation_tool,status="Not Executed"):
        super().__init__(test_id,test_name,module,status)
        self.automation_tool = automation_tool

    def display_test_case(self):
        super().display_test_case()
        print(f"Automation Tool: {self.automation_tool}")

    def to_csv_row(self):
        return [self.test_id, self.test_name, self.module, self.status, self.automation_tool]
    
class DocTestSuite:
    def __init__(self,suite_name):
        self.suite_name = suite_name
        self.test_cases = []

    def add_test_case(self,test_case):
        self.test_cases.append(test_case)

    def run_all_tests(self):
        print(f"Running Test Suite: {self.suite_name}")
        for test_case in self.test_cases:
            test_case.display_test_case()
            result = input("Enter test result (Passed/Failed): ").strip()
            test_case.execute_test(result)

    def save_results_to_csv(self,file_name):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Test ID", "Test Name", "Module", "Status", "Automation Tool"])
            for test_case in self.test_cases:
                writer.writerow(test_case.to_csv_row())
        print(f"Test results saved to {file_name}")

    def summary_report(self):
        total = len(self.test_cases)
        passed = sum(1 for test_case in self.test_cases if test_case.status == "Passed")
        failed = sum(1 for test_case in self.test_cases if test_case.status == "Failed")
        not_executed = total - (passed + failed)
        print("\n--- Test Suite Summary Report ---")
        print(f"Total Test Cases: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Not Executed: {not_executed}")
           
if __name__ == "__main__":
    tc1 = TestCase(1,"Login Test","Authentication")
    tc2 = AutomatedTestCase(2,"Signup Test","Authentication","Selenium")
    tc3 = TestCase(3,"Payment Test","Payment")
    tc4 = AutomatedTestCase(4,"Invoice Test","Payment","Postman")

    suite = DocTestSuite("User Management Tests")
    suite.add_test_case(tc1)
    suite.add_test_case(tc2)
    suite.add_test_case(tc3)
    suite.add_test_case(tc4)

    suite.run_all_tests()
    suite.summary_report()
    suite.save_results_to_csv("test_results.csv")