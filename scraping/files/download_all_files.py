import aiohttp
import asyncio
import os
import aiofiles
import random
import time
from urllib.parse import urlparse
from tqdm.asyncio import tqdm

DOWNLOAD_FOLDER = "downloaded_files"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # Start with 1 second wait time
MAX_CONCURRENCY = 50  # Reduce if needed to avoid rate limits

async def download_file(session, url, semaphore):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url) as response:
                    if response.status == 429:  # Too Many Requests
                        retry_after = int(response.headers.get("Retry-After", INITIAL_BACKOFF * (2 ** attempt)))
                        print(f"Rate limited. Retrying {url} in {retry_after} sec...")
                        await asyncio.sleep(retry_after + random.uniform(0.5, 1.5))  # Jitter
                        continue  # Retry the request
                    
                    response.raise_for_status()
                    parsed_url = urlparse(url).path.strip("/").split("/")
                    file_name = f"{parsed_url[0]}_{parsed_url[1]}_{parsed_url[2]}.py"

                    # Ensure unique filename
                    original_file_name = file_name
                    counter = 1
                    while os.path.exists(os.path.join(DOWNLOAD_FOLDER, file_name)):
                        file_name = f"{os.path.splitext(original_file_name)[0]}_{counter}{os.path.splitext(original_file_name)[1]}"
                        counter += 1

                    # Async file write
                    file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
                    async with aiofiles.open(file_path, "wb") as file:
                        await file.write(await response.read())

                    return f"✅ Downloaded {file_name}"

            except aiohttp.ClientResponseError as e:
                if response.status in {500, 502, 503, 504}:  # Server errors
                    print(f"Server error {response.status}. Retrying {url}...")
                    await asyncio.sleep(INITIAL_BACKOFF * (2 ** attempt))
                    continue
                else:
                    return f"❌ Failed {url}: {e}"

            except Exception as e:
                return f"❌ Failed {url}: {e}"

    return f"🚨 Gave up after {MAX_RETRIES} retries: {url}"

async def download_files_concurrently(urls, concurrency=MAX_CONCURRENCY):
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [download_file(session, url, semaphore) for url in urls]
        results = await tqdm.gather(*tasks, total=len(tasks))
        for result in results:
            print(result)

def read_urls_from_file(file_name):
    with open(file_name, "r") as file:
        return [line.strip() for line in file.readlines()]

# Run the async downloader
urls = read_urls_from_file("python_files.txt")
asyncio.run(download_files_concurrently(urls))
