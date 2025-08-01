import os
import glob
import pandas as pd

# Directory containing the CSV files
csv_dir = "runs_jsons/loss_epoch"

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

total_time = 0.0

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Assuming the time column is named 'time' or similar
    # Adjust the column name if needed
    time_col = None
    for col in df.columns:
        if "time" in col.lower():
            time_col = col
            break
    # print(df.columns)
    if time_col is None:
        continue  # Skip if no time column found
    first_time = df[time_col].iloc[0]
    last_time = df[time_col].iloc[-1]
    total_time += last_time - first_time


print(f"Cumulative time spent: {total_time} seconds")
hours = 1 / 3600
days = hours / 24
print(f"Cumulative time spent: {total_time * hours} hours")
print(f"Cumulative time spent: {total_time * days} days")
