#!/usr/bin/env python3
"""
Create metadata table for the elaborated GitHub Python dataset.

This script parses the python_files_elaborated.txt file containing GitHub URLs
and extracts repository metadata (owner, repo name, file path).
It generates a CSV file with this information and prints statistics.

The elaborated dataset contains more files than the licensed subset and
may include repositories with various licenses (not just permissive ones).
"""

import csv
import os
import re
from collections import Counter
from tqdm import tqdm

# Input and output files
ELABORATED_FILES_LIST = "python_files_elaborated.txt"
LICENSED_FILES_LIST = "python_files.txt"
OUTPUT_CSV = "python_files_elaborated_metadata.csv"

# Regular expression to parse GitHub raw URLs
# Format: https://raw.githubusercontent.com/OWNER/REPO/BRANCH/PATH
GITHUB_RAW_PATTERN = r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/[^/]+/(.*)"


def parse_github_url(url):
    """
    Parse a GitHub raw URL to extract owner, repo name, and file path.

    Args:
        url (str): GitHub raw URL

    Returns:
        tuple: (owner, repo_name, file_path) or None if URL doesn't match pattern
    """
    match = re.match(GITHUB_RAW_PATTERN, url)
    if match:
        owner, repo_name, file_path = match.groups()
        return owner, repo_name, file_path
    return None


def create_metadata_table(file_list_path):
    """
    Create a metadata table from a list of GitHub URLs.

    Args:
        file_list_path (str): Path to file containing GitHub URLs

    Returns:
        list: List of dictionaries with metadata
    """
    metadata = []

    # Read URLs from file
    with open(file_list_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Processing URLs from {file_list_path}...")

    # Parse each URL and extract metadata
    for url in tqdm(urls, desc="Parsing URLs"):
        parsed = parse_github_url(url)
        if parsed:
            owner, repo_name, file_path = parsed
            metadata.append(
                {
                    "owner": owner,
                    "repo_name": repo_name,
                    "file_path": file_path,
                    "url": url,
                }
            )

    return metadata


def generate_statistics(metadata, dataset_name):
    """
    Generate and print statistics for the dataset.

    Args:
        metadata (list): List of dictionaries with metadata
        dataset_name (str): Name of the dataset for display
    """
    # Count unique repositories and owners
    repos = set((item["owner"], item["repo_name"]) for item in metadata)
    owners = set(item["owner"] for item in metadata)

    # Count files by repository
    repo_counts = Counter((item["owner"], item["repo_name"]) for item in metadata)
    top_repos = repo_counts.most_common(10)

    # Count files by owner
    owner_counts = Counter(item["owner"] for item in metadata)
    top_owners = owner_counts.most_common(5)

    # Count file extensions
    extensions = Counter(os.path.splitext(item["file_path"])[1] for item in metadata)

    # Print statistics
    print(f"\n=== {dataset_name} Statistics ===")
    print(f"Total files: {len(metadata)}")
    print(f"Unique repositories: {len(repos)}")
    print(f"Unique repository owners: {len(owners)}")

    print("\nTop 10 repositories by file count:")
    for (owner, repo), count in top_repos:
        print(f"  {owner}/{repo}: {count} files")

    print("\nFile extensions:")
    for ext, count in extensions.most_common():
        if ext:  # Skip empty extensions
            print(f"  {ext}: {count} files")

    print("\nTop 5 repository owners:")
    for owner, count in top_owners:
        print(f"  {owner}: {count} files")

    return {
        "total_files": len(metadata),
        "unique_repos": len(repos),
        "unique_owners": len(owners),
        "top_repos": top_repos,
        "top_owners": top_owners,
        "extensions": extensions,
    }


def compare_datasets(elaborated_stats, licensed_stats):
    """
    Compare statistics between elaborated and licensed datasets.

    Args:
        elaborated_stats (dict): Statistics for elaborated dataset
        licensed_stats (dict): Statistics for licensed dataset
    """
    print("\n=== Dataset Comparison ===")
    print(f"Elaborated dataset: {elaborated_stats['total_files']} files")
    print(f"Licensed dataset: {licensed_stats['total_files']} files")
    print(
        f"Additional files in elaborated dataset: {elaborated_stats['total_files'] - licensed_stats['total_files']} files"
    )

    # Calculate percentage increase
    pct_increase = (
        (elaborated_stats["total_files"] / licensed_stats["total_files"]) - 1
    ) * 100
    print(f"Percentage increase: {pct_increase:.1f}%")

    # Compare repositories
    print(f"\nElaborated dataset: {elaborated_stats['unique_repos']} repositories")
    print(f"Licensed dataset: {licensed_stats['unique_repos']} repositories")

    # Compare owners
    print(
        f"\nElaborated dataset: {elaborated_stats['unique_owners']} repository owners"
    )
    print(f"Licensed dataset: {licensed_stats['unique_owners']} repository owners")

    # Find repositories unique to elaborated dataset
    elaborated_repos = set(
        (owner, repo) for (owner, repo), _ in elaborated_stats["top_repos"]
    )
    licensed_repos = set(
        (owner, repo) for (owner, repo), _ in licensed_stats["top_repos"]
    )
    unique_to_elaborated = elaborated_repos - licensed_repos

    if unique_to_elaborated:
        print("\nTop repositories unique to elaborated dataset:")
        for owner, repo in list(unique_to_elaborated)[:5]:
            print(f"  {owner}/{repo}")


def main():
    # Process elaborated dataset
    elaborated_metadata = create_metadata_table(ELABORATED_FILES_LIST)

    # Save to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["owner", "repo_name", "file_path", "url"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(elaborated_metadata)

    print(f"Metadata saved to {OUTPUT_CSV}")

    # Generate statistics for elaborated dataset
    elaborated_stats = generate_statistics(elaborated_metadata, "Elaborated Dataset")

    # Process licensed dataset for comparison
    if os.path.exists(LICENSED_FILES_LIST):
        licensed_metadata = create_metadata_table(LICENSED_FILES_LIST)
        licensed_stats = generate_statistics(licensed_metadata, "Licensed Dataset")

        # Compare datasets
        compare_datasets(elaborated_stats, licensed_stats)
    else:
        print(f"Warning: {LICENSED_FILES_LIST} not found. Cannot compare datasets.")


if __name__ == "__main__":
    main()
