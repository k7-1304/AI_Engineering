class BankAccount:
    def __init__(self, account_number, account_type, balance=0):
        self.account_number = account_number
        self.account_type = account_type
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited: {amount}. New balance: {self.balance}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew: {amount}. New balance: {self.balance}")
        else:
            print("Insufficient funds or invalid withdrawal amount.")

    def display_balance(self):
        print(f"Account Balance for {self.account_type}: {self.balance}")

if __name__ == "__main__":
    account1 = BankAccount("123456789", "Savings", 1000)
    account1.display_balance()
    account1.deposit(500)
    account1.withdraw(200)
    account1.withdraw(2000)
    account1.display_balance()