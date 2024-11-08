import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

# GitHub API setup
headers = {"Authorization": "token " + os.environ["GITHUB_PAT"]}
timeout_duration = 10  # Timeout for requests in seconds
output_file = "python_files.txt"

def fetch_python_files_from_repo(repo_url):
    # Get the repository name from the URL (e.g., "owner/repo" from "https://github.com/owner/repo")
    repo_name = repo_url.split("https://github.com/")[-1]

    # Fetch the contents of the repository
    contents_url = f"https://api.github.com/repos/{repo_name}/contents"
    response = requests.get(contents_url, headers=headers, timeout=timeout_duration)

    if response.status_code == 200:
        contents = response.json()
        
        # Loop through each file in the repository
        for file_data in contents:
            if file_data['name'].endswith('.py'):  # Python files
                file_size = file_data.get('size', 0)
                
                # Filter by file size (between 200 bytes and 1 MB)
                if 200 <= file_size <= 1000000:
                    with open(output_file, "a") as file:
                        file.write(f"{file_data['download_url']}\n")
                else:
                    print(f"Skipping {file_data['name']} (size {file_size} bytes)")
    else:
        print(f"Failed to fetch contents for {repo_name}: {response.status_code}")

def read_repositories_from_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()

# Read repository URLs from repositories.txt
repositories = read_repositories_from_file("repositories.txt")

# Fetch Python files from each repository
with tqdm(total=len(repositories), desc="Processing Repositories") as pbar:
    for repo_url in repositories:
        repo_url = repo_url.strip()
        if repo_url:  # Skip empty lines
            print(f"Fetching Python files from repository: {repo_url}")
            fetch_python_files_from_repo(repo_url)
        pbar.update(1)
        time.sleep(1)  # Delay to prevent rate limiting

print(f"Python files saved to {output_file}")
