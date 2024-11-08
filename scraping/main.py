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

def fetch_all_repositories():
    page = 1
    with open("repositories.txt", "w") as file:
        with tqdm(desc="Processing") as pbar:
            while True:
                params["page"] = page
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    repositories = data.get("items", [])
                    for repo in repositories:
                        file.write(f"{repo['html_url']}\n")
                    
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
