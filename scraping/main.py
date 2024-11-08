import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://api.github.com/search/repositories"
query = "language:Python size:5..5000 stars:>=100 template:false"
params = {
    "q": query,
    "per_page": 100
}
headers = {"Authorization":"token "+os.environ["GITHUB_PAT"]}

def get_existing_line_count(filename="repositories.txt"):
    if not os.path.exists(filename):
        return 0
    with open(filename, "r") as file:
        return sum(1 for _ in file)

def fetch_all_repositories():
    existing_lines = get_existing_line_count()
    page = (existing_lines // params["per_page"]) + 1  # Calculate the starting page

    with open("repositories.txt", "a") as file:  # Append mode to resume
        with tqdm(desc="Processing", initial=existing_lines // params["per_page"]) as pbar:
            while True:
                params["page"] = page
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    repositories = data.get("items", [])
                    
                    for repo in repositories:
                        file.write(f"{repo['html_url']}\n")
                    
                    # Break if fewer results were returned, meaning we're done
                    if len(repositories) < params["per_page"]:
                        break

                    page += 1
                    pbar.update(1)
                    time.sleep(5)

                elif response.status_code == 429:
                    reset_time = int(response.headers.get("x-ratelimit-reset", 0))
                    wait_time = max(reset_time - int(time.time()), 0)
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)

                else:
                    print("Error:", response.status_code, response.json())
                    break

fetch_all_repositories()