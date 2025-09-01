# Ask the user to enter withdrawal amount
amount = int(input("Enter withdrawal amount: "))

# Check if amount is divisible by 100
if amount % 100 == 0:
    print(f"Dispensing {amount}")
else:
    print("Invalid amount")
