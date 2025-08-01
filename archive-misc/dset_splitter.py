import os
import tqdm


# get through all files in inputted path, put the first 80% in one file concatenated and the last 20% in another
def split_files(input_path, output_path1, output_path2):
    files = [
        f
        for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f)) and f.endswith(".py")
    ]

    # Sort files to maintain order
    files.sort()

    split_index = int(len(files) * 0.8)
    first_80_files = files[:split_index]
    last_20_files = files[split_index:]

    print(os.listdir(input_path))

    with open(output_path1, "w") as outfile1:
        for fname in tqdm.tqdm(first_80_files):
            with open(os.path.join(input_path, fname), errors="ignore") as infile:
                outfile1.write(infile.read())
                outfile1.write("\nprint('---FILESEP---')\n")

    with open(output_path2, "w") as outfile2:
        for fname in tqdm.tqdm(last_20_files):
            with open(os.path.join(input_path, fname), errors="ignore") as infile:
                outfile2.write(infile.read())
                outfile2.write("\nprint('---FILESEP---')\n")


# Example usage
input_path = os.path.expanduser("~/torch_datasets/github-python/all_trains")
output_path1 = os.path.expanduser("~/torch_datasets/github-python/80")
output_path2 = os.path.expanduser("~/torch_datasets/github-python/20")
split_files(input_path, output_path1, output_path2)
