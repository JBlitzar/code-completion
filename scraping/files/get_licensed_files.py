import os
import glob
import shutil
from tqdm import tqdm

path = os.path.expanduser("~/torch_datasets/github-python/mega_corpus_all_files")
output_path = os.path.expanduser(
    "~/torch_datasets/github-python/mega_licensed_all_files"
)

with open("python_files.txt", "r") as f:
    all_urls = {line.strip() for line in f if line.strip()}

with open("python_files_allowed.txt", "r") as f:
    allowed_urls = {line.strip() for line in f if line.strip()}

# Find URLs in python_files_allowed.txt that are not in python_files.txt
missing_urls = allowed_urls - all_urls

if missing_urls:
    print(
        "The following URLs are in python_files_allowed.txt but not in python_files.txt:"
    )
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


with open("python_files_allowed.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip()]
    repo_paths = set(["/".join(url.split("//")[1].split("/")[1:3]) for url in urls])
    print(repo_paths)

    num_existing = 0
    all_files = glob.glob(os.path.join(path, "*.py"))

    for file in (pbar := tqdm(all_files)):
        if any(repo_path in file.replace("_", "/") for repo_path in repo_paths):
            num_existing += 1
            file_name = os.path.basename(file)
            shutil.copy(file, os.path.join(output_path, file_name))
            pbar.set_description(f"Copied {num_existing} files")

        else:
            # print(f"File not found: {file}")
            pass

    print(f"Number of existing files: {num_existing}")
