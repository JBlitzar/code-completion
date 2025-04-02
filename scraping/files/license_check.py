import requests
import os
from tqdm import tqdm

allowed_licenses = ["MIT License", "Apache", "BSD", "CC0", "fuck", "f*ck" "unlicense", "Public Domain","SOFTWARE IS PROVIDED"]
def check_license_from_direct_url(direct_url):
    resp = requests.get(direct_url)

    if resp.status_code == "200":
        text = resp.text
        for license in allowed_licenses:
            if license.lower() in text.lower():
                return True
    else:
        raise ValueError


def check_license_from_file_url(file_url):
    # example: https://raw.githubusercontent.com/ssloy/tinyoptimizer/main/analyzer.py

    base = "".join(file_url.split("/")[:-1])

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
    files = f.readlines()
    files = [f.strip() for f in files]
    for file in tqdm(files):
        if check_license_from_direct_url(file):
            allowed.append(file)


with open("python_files_allowed.txt", "w+") as f:
    for file in allowed:
        f.write(file + "\n")




