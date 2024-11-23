import os


def calculate_ascii_percentage(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        
        total_chars = len(data)
        if total_chars == 0:
            return 0

        ascii_chars = sum(1 for c in data if 0 <= c <= 127)
        percentage = (ascii_chars / total_chars) * 100

        return percentage
    except Exception as e:
        print(f"Error: {e}")
        return None

file_path = os.path.expanduser("~/torch_datasets/github-python/corpus/data/corpus_processed.txt")
ascii_percentage = calculate_ascii_percentage(file_path)
if ascii_percentage is not None:
    print(f"Percentage of ASCII characters: {ascii_percentage:.2f}%")


def find_unicode_passages(file_path, threshold=0.5, min_length=20):
    """
    Prints passages with a high density of non-ASCII characters.
    Args:
        file_path (str): Path to the input file.
        threshold (float): Proportion of non-ASCII characters to flag a line.
        min_length (int): Minimum length of a line to be considered.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, start=1):
                total_chars = len(line.strip())
                if total_chars < min_length:
                    continue  # Skip short lines
                
                non_ascii_count = sum(1 for c in line if ord(c) >= 128)
                if non_ascii_count / total_chars > threshold:
                    print(f"Line {line_num}: {line.strip()}")
                    print(f"  -> Non-ASCII Density: {non_ascii_count / total_chars:.2%}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
find_unicode_passages(file_path, threshold=0.5, min_length=20)

