import os
import re
import numpy as np

def read_and_compute_average(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        numbers = [int(num) for num in first_line.split()]
        return sum(numbers)

def main():
    # Regular expression to match the file pattern
    pattern = re.compile(r'^([a-z]+)_([A-Z]+)\.dat$')

    # Dictionary to hold the results
    results = {}

    # Search for .dat files in the current directory
    for filename in os.listdir('.'):
        match = pattern.match(filename)
        if match:
            detector, descriptor = match.groups()
            average = read_and_compute_average(filename)
            
            if detector not in results:
                results[detector] = {}
            results[detector][descriptor] = average

    # Extract unique descriptors for table columns
    descriptors = sorted({desc for det in results for desc in results[det]})

    # Create and print the Markdown table
    markdown_table = "| Detector | " + " | ".join(descriptors) + " |\n"
    markdown_table += "|---" * (len(descriptors) + 1) + "|\n"

    for detector, desc_values in results.items():
        row = [str(desc_values.get(desc, '')) for desc in descriptors]
        markdown_table += f"| {detector} | " + " | ".join(row) + " |\n"

    print(markdown_table)

if __name__ == "__main__":
    main()
