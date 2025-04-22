import requests
import os
from tqdm import tqdm
import time

allowed_licenses = [
    "MIT License",
    "MIT",
    "Apache License",
    "Apache 2.0",
    "Apache-2.0",
    "BSD License",
    "BSD",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0",
    "CC0 1.0",
    "Creative Commons Zero",
    "Creative Commons CC0",
    "CC0 Public Domain Dedication",
    "Unlicense",
    "The Unlicense",
    "Public Domain",
    "WTFPL",
    "DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE",
    "WTFPL License",
    "ISC",
    "ISC License",
    "Zlib",
    "Zlib License",
    "Boost Software License",
    "BSL-1.0",
    "Boost License",
    "Artistic License 2.0",
    "Python Software Foundation License",
    "PSF License",
    "SOFTWARE IS PROVIDED",
    "f*ck",
    "fuck",
]


allowed_repos = []


def check_license_from_direct_url(direct_url):
    resp = requests.get(direct_url)

    if resp.status_code == 200:
        text = resp.text
        for lic in allowed_licenses:
            if lic.lower() in text.lower():
                return True
    elif resp.status_code == 429:
        print("Rate-limited, waiting for 5m...")
        time.sleep(300)  # sleep for 5 minutes
        resp = requests.get(direct_url)
        if resp.status_code == 200:
            text = resp.text
            for lic in allowed_licenses:
                if lic.lower() in text.lower():
                    return True
        else:
            raise ValueError
    else:
        raise ValueError
    return False


def check_license_from_file_url(file_url):
    # example: https://raw.githubusercontent.com/ssloy/tinyoptimizer/main/analyzer.py
    base = "/".join(file_url.split("/")[:-1])
    suffixes = [
        "LICENSE",
        "license",
        "LICENSE.txt",
        "license.txt",
        "LICENSE.md",
        "license.md",
    ]
    if base in allowed_repos:
        return True
    for suffix in suffixes:
        try:
            if check_license_from_direct_url(f"{base}/{suffix}"):
                allowed_repos.append(base)
                return True
        except ValueError:
            continue
    return False


def get_last_line_number(file_path):
    """Reads the last processed line number."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def save_last_line_number(file_path, line_number):
    """Saves the last processed line number."""
    with open(file_path, "w") as f:
        f.write(str(line_number))


allowed_files = []
line_number_file = "license_line_number.txt"

with open("python_files.txt", "r") as f:
    files = [line.strip() for line in f if line.strip()]

last_line_number = get_last_line_number(line_number_file)
num_allowed = 0

# Use enumerate to track line numbers
for index, file in enumerate(
    tqdm(files[last_line_number:], initial=last_line_number, total=len(files)),
    start=last_line_number,
):
    try:
        if check_license_from_file_url(file):
            allowed_files.append(file)
            num_allowed += 1
    except Exception:
        pass
    # Update the progress description and save progress at each iteration
    tqdm.write(f"Allowed: {num_allowed}")
    save_last_line_number(line_number_file, index + 1)

    with open("python_files_allowed.txt", "w") as f:
        for file in allowed_files:
            f.write(file + "\n")
