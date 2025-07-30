from datasets import Dataset
import pandas as pd

a_path = "A_python_files_elaborated_metadata.csv"
b_path = "B_github_python_metadata.csv"
c_path = "C_mega_licensed_corpus_redacted.txt"

df_a = pd.read_csv(a_path)
df_a = df_a.rename(columns={
    "owner": "repo_owner",
    "url": "file_url"
})
ds_a = Dataset.from_pandas(df_a)

ds_a.push_to_hub("jblitzar/github-python-meta-elaborated", split="train")


df_b = pd.read_csv("B_github_python_metadata.csv")
df_b["repo_url"] = df_b.get("repo_url", "unknown")
print("Columns in df_b:", df_b.columns)
from datasets import Dataset
ds_b = Dataset.from_pandas(df_b)
ds_b.push_to_hub("jblitzar/github-python-metadata", split="train")


with open(c_path, encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()
ds_c = Dataset.from_dict({"text": lines})
ds_c.push_to_hub("jblitzar/github-python-corpus", split="train")
