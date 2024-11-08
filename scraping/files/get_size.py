import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

# Path to the file containing URLs
file_path = 'python_files.txt'

# Function to get the size of a file from a URL using a HEAD request
def get_file_size(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        # Extract the content length header
        size = int(response.headers.get('Content-Length', 0))
        return size
    except (requests.RequestException, ValueError):
        # Return 0 if any error occurs (e.g., timeout, invalid URL, or missing header)
        return 0

# Main function to calculate total size of all URLs in file
def calculate_total_size(file_path):
    # Read URLs from file
    if not os.path.exists(file_path):
        print("File not found!")
        return 0

    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]

    # Use threading to perform requests concurrently with a progress bar
    total_size = 0
    with ThreadPoolExecutor() as executor:
        # Wrap the map in tqdm for a progress bar
        file_sizes = list(tqdm(executor.map(get_file_size, urls), total=len(urls), desc=f"Processing URLs. Total so far: {sum(file_sizes)}"))

    # Calculate the total size
    total_size = sum(file_sizes)
    return total_size

# Calculate and print the total size
total_size = calculate_total_size(file_path)
print(f"Total size of all files: {total_size / (1024 * 1024):.2f} MB")
