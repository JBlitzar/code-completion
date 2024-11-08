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
sha_file = "seen_shas.txt"
line_number_file = "line_number.txt"

def fetch_python_files_from_repo(repo_url, seen_shas):
    # Get the repository name from the URL (e.g., "owner/repo" from "https://github.com/owner/repo")
    repo_name = repo_url.split("https://github.com/")[-1]

    # Fetch the contents of the repository
    contents_url = f"https://api.github.com/repos/{repo_name}/contents"
    response = requests.get(contents_url, headers=headers, timeout=timeout_duration)

    if response.status_code == 200:
        contents = response.json()
        
        # Loop through each file in the repository
        for file_data in contents:
            if file_data['name'].endswith('.py') and not file_data["name"].endswith("setup.py"):  # Python files
                file_size = file_data.get('size', 0)
                
                # Filter by file size (between 1 kb and 100 kb)
                if 1000 <= file_size <= 100000:
                    file_sha = file_data.get('sha')

                    # Skip if SHA has been seen before
                    if file_sha not in seen_shas:
                        with open(output_file, "a") as file:
                            file.write(f"{file_data['download_url']}\n")
                        seen_shas.add(file_sha)
                        # Also write the SHA to sha_file
                        with open(sha_file, "a") as sha_log:
                            sha_log.write(f"{file_sha}\n")
                    else:
                        print(f"Skipping file {file_data['name']} (SHA {file_sha} already seen)")
    else:
        print(f"Failed to fetch contents for {repo_name}: {response.status_code}")

def read_repositories_from_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()

def read_seen_shas(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(f.read().splitlines())
    return set()

def get_last_line_number(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return int(f.read().strip())
    return 0

def save_last_line_number(file_path, line_number):
    with open(file_path, "w") as f:
        f.write(str(line_number))

# Read repository URLs from repositories.txt
repositories = read_repositories_from_file("repositories.txt")

# Read previously seen SHAs from seen_shas.txt
seen_shas = read_seen_shas(sha_file)

# Read the last line number from line_number.txt to resume
last_line_number = get_last_line_number(line_number_file)

# Process repositories from the last saved line number
with tqdm(total=len(repositories) - last_line_number, desc="Processing Repositories") as pbar:
    for i in range(last_line_number, len(repositories)):
        repo_url = repositories[i].strip()
        if repo_url:  # Skip empty lines
            print(f"Fetching Python files from repository: {repo_url}")
            fetch_python_files_from_repo(repo_url, seen_shas)
            save_last_line_number(line_number_file, i + 1)  # Save the current line number for resume
        pbar.update(1)
        #time.sleep(0.1)  # Delay to prevent rate limiting

print(f"Python files saved to {output_file}")
