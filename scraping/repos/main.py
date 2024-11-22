import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

url = "https://api.github.com/search/repositories"
headers = {"Authorization": "token " + os.environ["GITHUB_PAT"]}
timeout_duration = 10  # Timeout for requests in seconds
output_file = "repositories.txt"
last_date_file = "last_date.txt"


def fetch_repositories_by_date_range(start_date, end_date):
    page = 1
    query = f"language:Python size:5..5000 stars:>=100 created:{start_date}..{end_date}"
    params = {
        "q": query,
        "per_page": 100,
        "sort": "stars",
    }

    with open(output_file, "a") as file, tqdm(
        desc=f"Fetching {start_date} to {end_date}", unit="page"
    ) as pbar:
        while True:
            params["page"] = page
            try:
                response = requests.get(
                    url, headers=headers, params=params, timeout=timeout_duration
                )

                if response.status_code == 200:
                    data = response.json()
                    repositories = data.get("items", [])

                    for repo in repositories:
                        file.write(f"{repo['html_url']}\n")

                    if len(repositories) < params["per_page"]:
                        break  # End if fewer results are returned

                    page += 1
                    pbar.update(1)
                    time.sleep(1)  # Adjust delay as needed

                elif response.status_code == 429:
                    reset_time = int(response.headers.get("x-ratelimit-reset", 0))
                    wait_time = max(reset_time - int(time.time()), 0)
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)

                else:
                    print("Error:", response.status_code, response.json())
                    break

            except requests.exceptions.Timeout:
                print("Request timed out. Retrying...")
                time.sleep(5)  # Delay before retrying on timeout


def generate_date_ranges(start_year=2015):
    end_date = datetime.now()
    current_date = datetime(start_year, 1, 1)

    if os.path.exists(last_date_file):
        with open(last_date_file, "r") as f:
            last_date_str = f.read().strip()
            if last_date_str:
                current_date = datetime.strptime(last_date_str, "%Y-%m-%d")

    while current_date < end_date:
        next_date = current_date + timedelta(days=30)  # Move by roughly one month
        yield current_date.strftime("%Y-%m-%d"), min(next_date, end_date).strftime(
            "%Y-%m-%d"
        )
        current_date = next_date


# Run the script across date ranges with progress tracking
date_ranges = list(generate_date_ranges())
with tqdm(total=len(date_ranges), desc="Total Date Ranges") as date_pbar:
    for start_date, end_date in date_ranges:
        print(f"Fetching repositories created between {start_date} and {end_date}")
        fetch_repositories_by_date_range(start_date, end_date)

        # Save last processed date
        with open(last_date_file, "w") as f:
            f.write(end_date)

        date_pbar.update(1)
