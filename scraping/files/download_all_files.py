import aiohttp
import asyncio
import os
from urllib.parse import urlparse
import aiofiles
from tqdm.asyncio import tqdm

DOWNLOAD_FOLDER = "downloaded_files"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

async def download_file(session, url, semaphore):
    async with semaphore:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                parsed_url = urlparse(url).path.strip("/").split("/")
                file_name = f"{parsed_url[0]}_{parsed_url[1]}_{parsed_url[2]}.py"

                # Ensure filename uniqueness
                original_file_name = file_name
                counter = 1
                while os.path.exists(os.path.join(DOWNLOAD_FOLDER, file_name)):
                    file_name = f"{os.path.splitext(original_file_name)[0]}_{counter}{os.path.splitext(original_file_name)[1]}"
                    counter += 1

                # Write file asynchronously
                file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
                async with aiofiles.open(file_path, "wb") as file:
                    await file.write(await response.read())

                return f"Downloaded {file_name}"

        except Exception as e:
            return f"Failed to download {url}: {e}"

async def download_files_concurrently(urls, concurrency=50):
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
asyncio.run(download_files_concurrently(urls, concurrency=100))  # Adjust concurrency if needed
