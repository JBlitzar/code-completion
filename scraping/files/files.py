import requests
import time
import os
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GitHub API setup
GITHUB_TOKEN = os.getenv("GITHUB_PAT_v2")  # Use a single token
timeout_duration = 10  # Timeout for requests in seconds
output_file = "python_files.txt"
sha_file = "seen_shas.txt"
line_number_file = "line_number.txt"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_python_files_from_repo(repo_url, seen_shas):
    """Fetches Python files from a given GitHub repository."""
    repo_name = repo_url.split("https://github.com/")[-1]
    contents_url = f"https://api.github.com/repos/{repo_name}/contents"
    
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(contents_url, headers=headers, timeout=timeout_duration)

    if response.status_code == 200:
        contents = response.json()
        for file_data in contents:
            if file_data["name"].endswith(".py") and not file_data["name"].endswith("setup.py"):
                file_size = file_data.get("size", 0)
                if 1000 <= file_size <= 100000:  # Filter by size
                    file_sha = file_data.get("sha")

                    if file_sha not in seen_shas:  # Avoid duplicates
                        with open(output_file, "a") as file:
                            file.write(f"{file_data['download_url']}\n")
                        with open(sha_file, "a") as sha_log:
                            sha_log.write(f"{file_sha}\n")
                        seen_shas.add(file_sha)
                    else:
                        logging.info(f"Skipping {file_data['name']} (SHA already seen)")
        return True  # Successfully processed repo

    elif response.status_code == 403:
        reset_time = int(response.headers.get("X-RateLimit-Reset", time.time()))  # Get reset time
        wait_time = reset_time - int(time.time())  # Seconds until reset
        logging.warning(f"Rate limit hit! Waiting {wait_time // 60} minutes until reset...")
        time.sleep(wait_time + 1)  # Sleep until reset
        return fetch_python_files_from_repo(repo_url, seen_shas)  # Retry after reset

    logging.error(f"Failed to fetch {repo_name}. Status: {response.status_code}")
    return False

def read_repositories_from_file(file_path):
    """Reads repository URLs from a file."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def read_seen_shas(file_path):
    """Reads previously seen SHAs from a file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(f.read().splitlines())
    return set()

def get_last_line_number(file_path):
    """Reads the last processed line number."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return int(f.read().strip())
    return 0

def save_last_line_number(file_path, line_number):
    """Saves the last processed line number."""
    with open(file_path, "w") as f:
        f.write(str(line_number))

# Main script execution
repositories = read_repositories_from_file("repositories.txt")
seen_shas = read_seen_shas(sha_file)
last_line_number = get_last_line_number(line_number_file)

with tqdm(total=len(repositories) - last_line_number, desc="Processing Repositories") as pbar:
    for index, repo in enumerate(repositories[last_line_number:], start=last_line_number):
        success = fetch_python_files_from_repo(repo, seen_shas)
        if not success:
            continue  # Skip failed repos
        
        pbar.update(1)
        save_last_line_number(line_number_file, index + 1)  # Save progress

logging.info(f"Python files saved to {output_file}")
