import os
from urllib.parse import urlparse
import glob
import shutil
from tqdm import tqdm

path = os.path.expanduser("~/torch_datasets/github-python/mega_corpus_all_files")
output_path = os.path.expanduser("~/torch_datasets/github-python/mega_licensed_all_files")

with open("python_files.txt", "r") as f:
    all_urls = {line.strip() for line in f if line.strip()}

with open("python_files_allowed.txt", "r") as f:
    allowed_urls = {line.strip() for line in f if line.strip()}

# Find URLs in python_files_allowed.txt that are not in python_files.txt
missing_urls = allowed_urls - all_urls

if missing_urls:
    print("The following URLs are in python_files_allowed.txt but not in python_files.txt:")
    for url in missing_urls:
        print(url)
else:
    print("All URLs in python_files_allowed.txt are contained in python_files.txt.")


# Rename all .py files in the input path to ensure they have a single .py extension
for root, _, files in tqdm(os.walk(path)):
    for file in files:
        if file.endswith(".py"):
            old_file_path = os.path.join(root, file)
            new_file_name = file.split(".py")[0] + ".py"
            new_file_path = os.path.join(root, new_file_name)
            if old_file_path != new_file_path:
                os.rename(old_file_path, new_file_path)
print("Renaming completed.")
def get_file_from_url(url, download_folder):
    parsed_url = urlparse(url).path.strip("/").split("/")
    file_name = f"{parsed_url[0]}_{parsed_url[1]}_{parsed_url[2]}.py"
    file_name = file_name.split(".py")[0] + ".py"

    return file_name

def get_fixed_file(file_name, all_files):
    if file_name in all_files:
        return file_name
    else:
        base_name = file_name.split(".py")[0]
        for file in all_files:
            if file.startswith(base_name):
                return file
    return "<none>.py"

with open("python_files_allowed.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip()]
    allowed_files = [get_file_from_url(url, path) for url in urls]

    num_existing = 0
    all_files = glob.glob(os.path.join(path, "*.py"))

    for file in tqdm(allowed_files):
        file = os.path.join(path, file)
        if os.path.exists(file):
            file_name = os.path.basename(file)
            shutil.copy(file, os.path.join(output_path, file_name))
            num_existing += 1
        elif os.path.exists(get_fixed_file(file, all_files)):
            file_name = os.path.basename(file)
            shutil.copy(get_fixed_file(file, all_files), os.path.join(output_path, file_name))
            num_existing += 1
        else:
            print(f"File not found: {file}")
    print(f"Number of existing files: {num_existing}")
    print(f"Retention percentage: {num_existing / len(allowed_files) * 100:.2f}%")
