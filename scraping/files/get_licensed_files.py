import os
from urllib.parse import urlparse
import glob
import shutil
from tqdm import tqdm
import random
path = os.path.expanduser("~/torch_datasets/github-python/mega_corpus_all_files")

output_path = os.path.expanduser("~/torch_datasets/github-python/mega_licensed_all_files")

def get_file_from_url(url):
    parsed_url = urlparse(url).path.strip("/").split("/")
    file_name = f"{parsed_url[0]}_{parsed_url[1]}_{parsed_url[2]}"
    print(file_name)
    return os.path.join(path, file_name)

with open("python_files_allowed.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip()]
    allowed_files = [get_file_from_url(url) for url in urls]
    file_list = glob.glob(os.path.join(path, "*.py"))
    random.shuffle(file_list)
    for file in tqdm(file_list):
        file_name = os.path.basename(file).split(".py")[0]
        print(file_name)
        if file_name in allowed_files:
            print("ding ding")
            shutil.copy(file, os.path.join(output_path, file_name))
    