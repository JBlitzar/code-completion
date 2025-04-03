import os
from urllib.parse import urlparse
import glob
import shutil
from tqdm import tqdm

path = os.path.expanduser("~/torch_datasets/github-python/mega_corpus_all_files")
output_path = os.path.expanduser("~/torch_datasets/github-python/mega_licensed_all_files")

def get_file_from_url(url):
    parsed_url = urlparse(url).path.strip("/").split("/")
    file_name = f"{parsed_url[0]}_{parsed_url[1]}_{parsed_url[2]}.py"
    return os.path.join(path, file_name)

with open("python_files_allowed.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip()]
    allowed_files = [get_file_from_url(url) for url in urls]

    num_existing = 0

    for file in tqdm(allowed_files):
        if os.path.exists(file):
            file_name = os.path.basename(file)
            shutil.copy(file, os.path.join(output_path, file_name))
            num_existing += 1
        else:
            print(f"File not found: {file}")
    print(f"Number of existing files: {num_existing}")
    print(f"Retention percentage: {num_existing / len(allowed_files) * 100:.2f}%")
