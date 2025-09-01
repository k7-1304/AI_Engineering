n = int(input("Enter a integer: "))
def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(str(i))
if n > 0:
        fizzbuzz(n)
else:
        print("Please enter a number greater than 0.")

