import os


def split_file(input_file, lines_per_split=20):
    # Read the entire content of the file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Split the lines into chunks of the specified size
    chunks = [
        lines[i : i + lines_per_split] for i in range(0, len(lines), lines_per_split)
    ]

    # Save each chunk as a new file
    base_name = os.path.splitext(input_file)[0]
    for idx, chunk in enumerate(chunks):
        new_file_name = f"{base_name}_part{idx + 1}.txt"
        with open(new_file_name, "w") as f:
            f.writelines(chunk)

    # Delete the original file
    os.remove(input_file)


def split_all_files_in_directory(directory):
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Process only txt files
        if filename.endswith(".txt"):
            split_file(file_path)


# Example usage
directory = "."
split_all_files_in_directory(directory)
