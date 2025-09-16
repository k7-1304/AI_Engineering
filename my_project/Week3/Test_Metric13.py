import numpy as np

#np.random.seed(42)  # For reproducibility
data = np.random.randint(5, 51, size=(5,50))     # Random integers
print("Execution data (5 cycles × 50 tests):\n", data)

avg_per_cycle = np.mean(data, axis=1)
print("\nAverage execution time per cycle:", avg_per_cycle)

max_time = np.max(data)
max_loc = np.unravel_index(np.argmax(data), data.shape)
print("Max execution time:", max_time, "at Cycle:", max_loc[0]+1, "Test:", max_loc[1]+1)

std_dev = np.std(data, axis=1)
print("Standard deviation of execution times per cycle:", std_dev)

print("First 10 Cycle 1 Execution:", data[0, :10])
print("Last 5 execution times of Cycle 5:", data[4, -5:])
print("Every aletrnate test in Cycle 3:", data[2, ::2])

add_cycle = data[0] + data[1]
print("Sum of Cycle 1 and Cycle 2 execution times:", add_cycle)
sub_cycle = data[0] - data[1]
print("Difference of Cycle 1 and Cycle 2 execution times:", sub_cycle)
mul_cycle = data[3] * data[4]
print("Product of Cycle 4 and Cycle 5 execution times:", mul_cycle)
div_cycle = data[3] / data[4]
print("Division of Cycle 4 by Cycle 5 execution times:", div_cycle)


squared = np.square(data)
print("Squared execution times:\n", squared)
sqrted = np.sqrt(data)
print("Square root of execution times:\n", sqrted)
loged = np.log(data)
print("Natural log of execution times:\n", loged)
cubed = np.power(data, 3)
print("Cubed execution times:\n", cubed)

shallow_copy = data.view()
shallow_copy[0, :5] = 99
print("\nAfter modifying shallow copy (Cycle 1 first 5):")
print("Original Data (changed):\n", data)

deep_copy = data.copy()
deep_copy[1, :5] = 88
print("\nAfter modifying deep copy (Cycle 2 first 5):")
print("Deep Copy:\n", deep_copy)
print("Original Data (unchanged):\n", data)