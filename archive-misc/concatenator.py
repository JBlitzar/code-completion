import os
import glob
from tqdm import tqdm

folder = os.path.expanduser("~/torch_datasets/github-python/mega_licensed_all_files")
output_file = os.path.expanduser(
    "~/torch_datasets/github-python/mega_licensed_corpus/concatenated.py"
)

with open(output_file, "w", encoding="utf-8") as out_f:
    for file in tqdm(glob.glob(os.path.join(folder, "*.py"))):
        out_f.write("\n# <FILESEP>\n")
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as in_f:
                out_f.write(in_f.read())
        except Exception as e:
            out_f.write(f"\n# Skipping {file} due to error: {e}\n")

print(f"Concatenation complete: {output_file}")
