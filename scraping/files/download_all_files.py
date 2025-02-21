import requests
import os
from urllib.parse import urlparse
from tqdm import tqdm
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_file(url, session, download_folder):
    try:
        # Download the file from the URL
        response = session.get(url)
        response.raise_for_status()  # Check for errors

        # Parse the URL to extract the file name
        parsed_url = urlparse(url).path.strip("/").split("/")
        file_name = f"{parsed_url[0]}_{parsed_url[1]}_{parsed_url[2]}.py"

        # Ensure the file name is unique within the download folder
        original_file_name = file_name
        counter = 1
        while os.path.exists(os.path.join(download_folder, file_name)):
            # If the file name exists, append a counter to make it unique
            file_name = f"{os.path.splitext(original_file_name)[0]}_{counter}{os.path.splitext(original_file_name)[1]}"
            counter += 1

        # Write the file to the download folder
        with open(os.path.join(download_folder, file_name), "wb") as file:
            file.write(response.content)
        return f"Downloaded {file_name}"

    except requests.exceptions.RequestException as e:
        return f"Failed to download {url}: {e}"


def download_files_concurrently(urls, download_folder):
    # Ensure the download folder exists
    os.makedirs(download_folder, exist_ok=True)

    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit download tasks to the executor
            futures = [executor.submit(download_file, url, session, download_folder) for url in urls]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                #print(future.result())
                pass


def read_urls_from_file(file_name):
    # Read URLs from a text file and return as a list
    with open(file_name, "r") as file:
        # Strip newline characters and return as a list
        return [line.strip() for line in file.readlines()]


# Path to the text file containing URLs
urls_file = "python_files.txt"

# Folder to save the downloaded files
download_folder = "downloaded_files"

# Read the URLs from the file
urls = read_urls_from_file(urls_file)

# Download files concurrently and save them to the folder
download_files_concurrently(urls, download_folder)
