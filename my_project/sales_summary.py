sales = [1200, 3400, 560, 4500, 2100]
total_sales = sum(sales)
average_sales = total_sales / len(sales)
max_sales = max(sales)
min_sales = min(sales)
print(f"Total Sales: ${total_sales}")
print(f"Average Sales: ${average_sales:.2f}")
print(f"Highest Sale: ${max_sales}")
print(f"Lowest Sale: ${min_sales}") 