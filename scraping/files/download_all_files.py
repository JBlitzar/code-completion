import zipfile
import requests
import os
from urllib.parse import urlparse
from tqdm import tqdm

def download_and_add_to_zip(urls, zip_name):
    # Open the ZIP file in append mode (to add files incrementally)
    with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zipf:
        existing_files = {name for name in zipf.namelist()}  # Set of existing file names in the ZIP archive
        
        for url in tqdm(urls):
            try:
                # Download the file from the URL
                response = requests.get(url)
                response.raise_for_status()  # Check for errors

                # Parse the URL to extract the file name
                parsed_url = urlparse(url).path.strip("/").split("/")
                file_name = f"{parsed_url[0]}_{parsed_url[1]}_{parsed_url[2]}.py"

                # Ensure the file name is unique within the ZIP archive
                original_file_name = file_name
                counter = 1
                while file_name in existing_files:
                    # If the file name exists, append a counter to make it unique
                    file_name = f"{os.path.splitext(original_file_name)[0]}_{counter}{os.path.splitext(original_file_name)[1]}"
                    counter += 1

                # Write the file to the ZIP archive
                zipf.writestr(file_name, response.content)
                existing_files.add(file_name)  # Track the new file in the existing files set
                #print(f"Added {file_name} to the ZIP archive.")
                
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {url}: {e}")

def read_urls_from_file(file_name):
    # Read URLs from a text file and return as a list
    with open(file_name, 'r') as file:
        # Strip newline characters and return as a list
        return [line.strip() for line in file.readlines()]

# Path to the text file containing URLs
urls_file = "python_files.txt"

# Name of the ZIP file
zip_name = "corpus.zip"

# Read the URLs from the file
urls = read_urls_from_file(urls_file)

# Download and add each file to the ZIP archive incrementally
download_and_add_to_zip(urls, zip_name)
