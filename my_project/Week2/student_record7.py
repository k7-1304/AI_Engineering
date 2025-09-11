class student:
    def __init__(self,name, grade, department):
        self.name = name
        self.grade = grade
        self.department = department
    def print_info(self):
        print(f"Name: {self.name}, Grade: {self.grade}, Department: {self.department}")

    def update_grade(self,new_grade):
        self.grade = new_grade
        print(f"{self.name}'s grade updated to {self.grade}")

if __name__ == "__main__":

    student1 = student("Alice", "A", "Computer Science")
    student2 = student("Bob", "B", "Mathematics")
    student3 = student("Charlie", "A", "Physics")

student1.print_info()
student2.print_info()
student3.print_info()

student2.update_grade("A+")

student1.print_info()
student2.print_info()
student3.print_info()