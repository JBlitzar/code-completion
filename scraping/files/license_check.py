import requests
import os
from tqdm import tqdm
import concurrent.futures

allowed_licenses = [
    "MIT License", "MIT",
    "Apache License", "Apache 2.0", "Apache-2.0",
    "BSD License", "BSD", "BSD-2-Clause", "BSD-3-Clause",
    "CC0", "CC0 1.0", "Creative Commons Zero", "Creative Commons CC0", "CC0 Public Domain Dedication",
    "Unlicense", "The Unlicense", "Public Domain",
    "WTFPL", "DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE", "WTFPL License",
    "ISC", "ISC License",
    "Zlib", "Zlib License",
    "Boost Software License", "BSL-1.0", "Boost License",
    "Artistic License 2.0",
    "Python Software Foundation License", "PSF License",
    "SOFTWARE IS PROVIDED", "f*ck", "fuck"
]

def check_license_from_direct_url(direct_url):
    resp = requests.get(direct_url)
    if resp.status_code == 200:
        text = resp.text
        for license in allowed_licenses:
            if license.lower() in text.lower():
                return True
    else:
        raise ValueError(f"HTTP {resp.status_code} for {direct_url}")
    return False

def check_license_from_file_url(file_url):
    # example: https://raw.githubusercontent.com/ssloy/tinyoptimizer/main/analyzer.py
    base = "/".join(file_url.split("/")[:-1])
    suffixes = ["LICENSE", "license", "LICENSE.txt", "license.txt", "LICENSE.md", "license.md"]
    for suffix in suffixes:
        try:
            if check_license_from_direct_url(f"{base}/{suffix}"):
                return True
        except ValueError:
            continue
    return False

allowed = []

with open("python_files.txt", "r") as f:
    files = [line.strip() for line in f.readlines()]

# Adjust number of workers as needed.
max_workers = 10

# Use multithreading to process files concurrently.
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Map each URL to a future.
    future_to_url = {executor.submit(check_license_from_direct_url, file): file for file in files}
    # Initialize progress bar with the total number of files.
    num_allowed = 0
    for future in (pbar := tqdm(concurrent.futures.as_completed(future_to_url), total=len(files))):
        url = future_to_url[future]
        try:
            result = future.result()
            if result:
                allowed.append(url)
                num_allowed += 1
            pbar.set_description(f"Allowed: {num_allowed}")
        except Exception:
            # Ignore exceptions for now.
            continue

with open("python_files_allowed.txt", "w+") as f:
    for file in allowed:
        f.write(file + "\n")
