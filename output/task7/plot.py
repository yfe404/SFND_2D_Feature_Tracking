import matplotlib.pyplot as plt
import numpy as np
import sys

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = list(map(float, line.split()))
            data.append(numbers)
    return data

def plot_data(data):
    all_values = [val for sublist in data for val in sublist]
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(all_values, bins=20, color='blue', alpha=0.7)
    plt.title('Distribution of keypoint sizes')
    plt.xlabel('Size')
    plt.ylabel('Frequency')

    values_per_line = [len(sublist) for sublist in data]
    plt.subplot(1, 2, 2)
    plt.bar(range(len(values_per_line)), values_per_line, color='green', alpha=0.7)
    plt.title('Number of keypoints per images')
    plt.xlabel('Image number')
    plt.ylabel('Number of Keypoints')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py data_file.dat")
        sys.exit(1)

    file_path = sys.argv[1]
    data = read_data(file_path)
    plot_data(data)
