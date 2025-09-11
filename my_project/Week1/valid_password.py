# Predefined correct password
correct_password = "openAI123"

# Allow the user 3 attempts
for attempt in range(1, 4):
    entered_password = input(f"Attempt {attempt}/3 - Enter your password: ")

    if entered_password == correct_password:
        print("Login Successful")
        break
    else:
        print("Incorrect password.")

# If all 3 attempts fail → lock the account
else:
    print("Account Locked")
