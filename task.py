import random
import time
import matplotlib.pyplot as plt

# Sorting Algorithms


# Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


# Insertion Sort
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# Selection Sort
def selection_sort(arr):
    for i in range(len(arr)):
        max_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] > arr[max_idx]:
                max_idx = j
        arr[i], arr[max_idx] = arr[max_idx], arr[i]


# count sort
def count_sort(arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)
    output = [0] * len(arr)

    # Store the count of each element
    for num in arr:
        count[num] += 1

    # Modify count array by adding the previous counts
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Place elements into their correct position
    for num in reversed(arr):
        output[count[num] - 1] = num
        count[num] -= 1

    # Copy sorted elements back to original array
    for i in range(len(arr)):
        arr[i] = output[i]


# Radix sort
def count_sort_for_radix(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10  

    # Store count of occurrences for the digit at exp place
    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    # Change count so that count[i] contains the actual position of this digit
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array using the count array
    for i in reversed(range(n)):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    # Copy the output array to arr
    for i in range(n):
        arr[i] = output[i]


# Radix Sort main function
def radix_sort(arr):
    max_val = max(arr)

    # Apply count sort to each digit (exp is 10^i where i is the current digit)
    exp = 1
    while max_val // exp > 0:
        count_sort_for_radix(arr, exp)
        exp *= 10


# bucket sort
def bucket_sort(arr):
    max_value = max(arr)
    bucket_count = len(arr)
    buckets = [[] for _ in range(bucket_count)]

    # Put array elements into different buckets after normalizing the values
    for num in arr:
        index = min(
            int((num / max_value) * (bucket_count - 1)), bucket_count - 1
        )  # Ensure index is valid
        buckets[index].append(num)

    # Sort individual buckets and concatenate the result
    for bucket in buckets:
        insertion_sort(bucket)  

    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    for i in range(len(arr)):
        arr[i] = sorted_arr[i]


# Merge Sort - Uses Divide and Conquer


def merge_sort(arr):
    if len(arr) > 1:
        mid_index = len(arr) // 2

        left_part = arr[:mid_index]
        right_part = arr[mid_index:]

        merge_sort(left_part)
        merge_sort(right_part)

        left_idx = 0
        right_idx = 0
        main_idx = 0

        while left_idx < len(left_part) and right_idx < len(right_part):
            if left_part[left_idx] < right_part[right_idx]:
                arr[main_idx] = left_part[left_idx]
                left_idx += 1
            else:
                arr[main_idx] = right_part[right_idx]
                right_idx += 1
            main_idx += 1

        while left_idx < len(left_part):
            arr[main_idx] = left_part[left_idx]
            left_idx += 1
            main_idx += 1

        while right_idx < len(right_part):
            arr[main_idx] = right_part[right_idx]
            right_idx += 1
            main_idx += 1
    return arr


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)


def heapify(arr, n, i):
    # Initialize largest as root
    largest = i
    left = 2 * i + 1  # Left child
    right = 2 * i + 2  # Right child

    # Check if left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Check if right child exists and is greater than the current largest
    if right < n and arr[right] > arr[largest]:
        largest = right

    # If largest is not root, swap with the largest element
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        # Recursively heapify the affected subtree
        heapify(arr, n, largest)


def heap_sort(arr):
    n = len(arr)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        # Swap the current root (largest element) with the end of the array
        arr[i], arr[0] = arr[0], arr[i]
        # Heapify the reduced heap
        heapify(arr, i, 0)

    return arr


# generating a descending sorted random array
def generate_descending_array(size):
    arr = random.sample(range(1, size + 1), size)
    arr.sort(reverse=True)
    return arr


# measure execution time for each sorting algorithm
def measure_time(sorting_function, arr):
    start_time = time.time()
    sorting_function(arr)
    end_time = time.time()

    # Converting into milliseconds
    return (end_time - start_time) * 1000


# Array sizes to test
array_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]

# Initialize table to store results
table = {size: {} for size in array_sizes}

# Testing each sorting algorithm for different array sizes
for size in array_sizes:
    # print(f"\nGenerating array of size {size} (in descending order)...")

    #  Pre-sorted in descending order
    arr = generate_descending_array(size)

    print(f"Running Bubble Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Bubble"] = measure_time(bubble_sort, arr_copy)
    print(f"Completed Bubble Sort for array size {size}.")

    print(f"Running Insertion Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Insertion"] = measure_time(insertion_sort, arr_copy)
    print(f"Completed Insertion Sort for array size {size}.")

    print(f"Running Selection Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Selection"] = measure_time(selection_sort, arr_copy)
    print(f"Completed Selection Sort for array size {size}.")

    print(f"Running Merge Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Merge Sort"] = measure_time(merge_sort, arr_copy)

    print(f"Running Quick Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Quick Sort"] = measure_time(quick_sort, arr_copy)

    print(f"Running Quick Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Heap Sort"] = measure_time(heap_sort, arr_copy)

    print(f"Running Count Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Count Sort"] = measure_time(count_sort, arr_copy)

    print(f"Running Radix Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Radix Sort"] = measure_time(radix_sort, arr_copy)

    print(f"Running Bucket Sort for array size {size}...")
    arr_copy = arr.copy()
    table[size]["Bucket Sort"] = measure_time(bucket_sort, arr_copy)

# Display the results in a properly formatted table with consistent width
print(
    f"\n{'Array Size':<10} {'Bubble Sort (ms)':<20} {'Insertion Sort (ms)':<20} {'Selection Sort (ms)':<20} {'Count Sort (ms)':<20} {'Radix Sort (ms)':<20} {'Bucket Sort (ms)':<20} {'Merge Sort (ms)':<20} {'Quick Sort (ms)':<20} {'Heap Sort (ms)':<20}"
)
print("-" * 180)

for size in array_sizes:
    print(
        f"{size:<10} {table[size]['Bubble']:<20.2f} {table[size]['Insertion']:<20.2f} {table[size]['Selection']:<20.2f} {table[size]['Count Sort']:<20.2f} {table[size]['Radix Sort']:<20.2f} {table[size]['Bucket Sort']:<20.2f} {table[size]['Merge Sort']:<20.2f} {table[size]['Quick Sort']:<20.2f} {table[size]['Heap Sort']:<20.2f}"
    )

# Convert table data for plotting
bubble_times = [table[size]["Bubble"] for size in array_sizes]
insertion_times = [table[size]["Insertion"] for size in array_sizes]
selection_times = [table[size]["Selection"] for size in array_sizes]
mergetimes = [table[size]["Merge Sort"] for size in array_sizes]
heaptimes = [table[size]["Heap Sort"] for size in array_sizes]
quicktimes = [table[size]["Quick Sort"] for size in array_sizes]
count_times = [table[size]["Count Sort"] for size in array_sizes]
radix_times = [table[size]["Radix Sort"] for size in array_sizes]
bucket_times = [table[size]["Bucket Sort"] for size in array_sizes]

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot the first three sorts
plt.plot(array_sizes, bubble_times, marker="o", label="Bubble Sort", color="orange")
plt.plot(array_sizes, insertion_times, marker="o", label="Insertion Sort", color="blue")
plt.plot(array_sizes, selection_times, marker="o", label="Selection Sort", color="green")

# Plot Merge, Heap, and Quick Sort
plt.plot(array_sizes, mergetimes, marker="o", label="Merge Sort", color="teal")
plt.plot(array_sizes, heaptimes, marker="o", label="Heap Sort", color="pink")
plt.plot(array_sizes, quicktimes, marker="o", label="Quick Sort", color="yellow")

# Plot Count, Radix, and Bucket Sort
plt.plot(array_sizes, count_times, marker="o", label="Count Sort", color="red")
plt.plot(array_sizes, radix_times, marker="o", label="Radix Sort", color="purple")
plt.plot(array_sizes, bucket_times, marker="o", label="Bucket Sort", color="brown")

# Logarithmic scale for x-axis (Array sizes)
plt.xscale("log")

# Logarithmic scale for y-axis (Time in milliseconds)
plt.yscale("log")

# Labeling the plot
plt.xlabel("Array Size (log scale)")
plt.ylabel("Time (ms, log scale)")
plt.title("Empirical Time Complexity of Sorting Algorithms")

# Adding legend and grid
plt.legend()
plt.grid(True, which="both", ls="--")

# Show the plot
plt.show()
