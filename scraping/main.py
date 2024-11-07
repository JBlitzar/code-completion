import requests
import time
from tqdm import tqdm

url = "https://api.github.com/search/repositories"
query = "language:Python size:5..5000 stars:>=100 template:false"
params = {
    "q": query,
    "per_page": 100
}
headers = {"Authorization":"no peeking"}

response = requests.get("https://api.github.com/rate_limit", headers=headers)
rate_limit_data = response.json()
print(rate_limit_data)
def fetch_all_repositories():
    page = 1
    with open("repositories.txt", "w") as file:
        with tqdm(desc="Processing") as pbar:
            
            while True:
                pbar.update(1)
                params["page"] = page
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    print(response.headers["x-ratelimit-reset"])
                    data = response.json()
                    repositories = data.get("items", [])
                    for repo in repositories:
                        file.write(f"{repo['html_url']}\n")
                    if len(repositories) < params["per_page"]:
                        break
                    page += 1
                    time.sleep(2)

                else:
                    print("Error:", response.status_code, response.json())
                    break


fetch_all_repositories()
