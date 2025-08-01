#!/usr/bin/env python3
"""
Create a metadata table from GitHub Python file URLs.

This script processes the file URLs from python_files.txt and creates a tabular
CSV file with repository metadata including owner, name, file path, and URLs.
"""

import os
import re
import csv
import pandas as pd
from collections import Counter
from urllib.parse import urlparse
from tqdm import tqdm


def parse_github_url(url):
    """
    Parse a GitHub URL to extract repository owner, name, and file path.
    
    Handles both raw.githubusercontent.com and github.com URLs.
    
    Args:
        url (str): GitHub URL
        
    Returns:
        dict: Dictionary with repo_owner, repo_name, file_path, repo_url
    """
    url = url.strip()
    
    # Initialize default values
    result = {
        "repo_owner": "unknown",
        "repo_name": "unknown",
        "file_path": "",
        "file_url": url,
        "repo_url": ""
    }
    
    try:
        # Parse URL to get components
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        # Handle raw.githubusercontent.com URLs
        # Format: https://raw.githubusercontent.com/owner/repo/branch/path/to/file.py
        if 'raw.githubusercontent.com' in url:
            if len(path_parts) >= 3:
                result["repo_owner"] = path_parts[0]
                result["repo_name"] = path_parts[1]
                # Skip branch (path_parts[2]) and get the rest as file path
                result["file_path"] = '/'.join(path_parts[3:])
                result["repo_url"] = f"https://github.com/{path_parts[0]}/{path_parts[1]}"
        
        # Handle github.com URLs
        # Format: https://github.com/owner/repo/blob/branch/path/to/file.py
        elif 'github.com' in url:
            if len(path_parts) >= 4 and path_parts[2] == 'blob':
                result["repo_owner"] = path_parts[0]
                result["repo_name"] = path_parts[1]
                # Skip 'blob' and branch, get the rest as file path
                result["file_path"] = '/'.join(path_parts[4:])
                result["repo_url"] = f"https://github.com/{path_parts[0]}/{path_parts[1]}"
        
        return result
    
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        return result


def process_file_urls(input_file, output_file):
    """
    Process GitHub file URLs and create a metadata CSV file.
    
    Args:
        input_file (str): Path to the file containing GitHub URLs
        output_file (str): Path to the output CSV file
    """
    print(f"Processing URLs from {input_file}...")
    
    # Read file URLs
    with open(input_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Parse each URL
    metadata = []
    for url in tqdm(urls, desc="Parsing URLs"):
        metadata.append(parse_github_url(url))
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata)
    
    # Save to CSV
    # Use minimal quoting to remain compatible with the standard csv module
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Metadata saved to {output_file}")
    
    # Print statistics
    unique_repos = df[['repo_owner', 'repo_name']].drop_duplicates()
    unique_owners = df['repo_owner'].nunique()
    
    print("\n=== Dataset Statistics ===")
    print(f"Total files: {len(df)}")
    print(f"Unique repositories: {len(unique_repos)}")
    print(f"Unique repository owners: {unique_owners}")
    
    # Top repositories by file count
    repo_counts = Counter(zip(df['repo_owner'], df['repo_name']))
    print("\nTop 10 repositories by file count:")
    for (owner, repo), count in repo_counts.most_common(10):
        print(f"  {owner}/{repo}: {count} files")
    
    # File extensions
    extensions = Counter([os.path.splitext(path)[1] for path in df['file_path'] if path])
    print("\nFile extensions:")
    for ext, count in extensions.most_common(5):
        print(f"  {ext or 'No extension'}: {count} files")
    
    # Repository owners with most repositories
    owner_repo_counts = Counter(df['repo_owner'])
    print("\nTop 5 repository owners:")
    for owner, count in owner_repo_counts.most_common(5):
        print(f"  {owner}: {count} files")


if __name__ == "__main__":
    input_file = "python_files.txt"
    output_file = "github_python_metadata.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        print("Please make sure the file exists in the current directory.")
        exit(1)
    
    process_file_urls(input_file, output_file)
