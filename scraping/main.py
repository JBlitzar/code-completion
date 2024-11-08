import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

url = "https://api.github.com/search/repositories"
query = "language:Python size:5..5000 stars:>=100 template:false"
params = {
    "q": query,
    "per_page": 100
}
headers = {"Authorization":"token "+os.environ["GITHUB_PAT"]}
def fetch_repositories_by_date_range(start_date, end_date):
    page = 1
    query = f"language:Python size:5..5000 stars:>=100 created:{start_date}..{end_date}"
    params = {
        "q": query,
        "per_page": 100,
        "sort": "stars",
    }
    
    with open("repositories.txt", "a") as file:  # Append mode
        while True:
            params["page"] = page
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                repositories = data.get("items", [])
                
                for repo in repositories:
                    file.write(f"{repo['html_url']}\n")
                
                if len(repositories) < params["per_page"]:
                    break  # End if fewer results are returned

                page += 1
                time.sleep(1)  # Adjust delay as needed

            elif response.status_code == 429:
                reset_time = int(response.headers.get("x-ratelimit-reset", 0))
                wait_time = max(reset_time - int(time.time()), 0)
                print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)

            else:
                print("Error:", response.status_code, response.json())
                break

def generate_date_ranges(start_year=2015):
    end_date = datetime.now()
    current_date = datetime(start_year, 1, 1)
    
    while current_date < end_date:
        next_date = current_date + timedelta(days=30)  # Move by roughly one month
        yield current_date.strftime("%Y-%m-%d"), min(next_date, end_date).strftime("%Y-%m-%d")
        current_date = next_date

# Run the script across date ranges
for start_date, end_date in generate_date_ranges():
    print(f"Fetching repositories created between {start_date} and {end_date}")
    fetch_repositories_by_date_range(start_date, end_date)