---
annotations_creators:
  - author
license:
  - gpl-3.0
multilinguality:
  - monolingual
pretty_name: GitHub-Python
dataset_name: github-python
dataset_type: code
tags:
  - code
  - python
  - code-generation
size_categories:
  - 100K<n⩽1M
task_categories:
  - text-generation
task_ids:
  - code-completion
---

# GitHub-Python — Licensed & Elaborated Variants

This repository ships **two complementary Python-code corpora** extracted from
public GitHub:

* **Licensed Subset** – strictly *permissive-licensed* files suitable for
  commercial redistribution / model training (main corpus used in our
  experiments).
* **Elaborated Collection** – a broader crawl that additionally contains files
  under *copyleft* or unclear licenses (GPL/AGPL/LGPL, etc.).  Useful for
  analysis or pre-training where license mixing is acceptable.

Both variants target **code-completion / generation** research.

## Dataset at a glance

|                           | **Licensed Subset** | **Elaborated Collection** |
| ------------------------- | ------------------- | ------------------------- |
| Files (.py)               | 53,017              | 186,066                  |
| Unique repositories       | 16,447              | 59,852                   |
| Repository owners         | 12,515              | 43,517                   |
| Compressed size           | 732 MB              | 2.4 GB \*               |
| Vocabulary (tokens)       | 443,431             | 443,431 †               |
| License coverage          | Permissive only     | Mixed (perm. + copyleft) |
| Secrets redacted          | ✅                  | ⚠️ not guaranteed       |
| Time window               | ≥ 2015-01-01        | ≥ 2015-01-01             |

\* estimated – elaborated corpus is distributed as raw file list, not a single
text file.  
† same tokenizer file is shared by both variants.

Numbers were obtained from the final redacted corpus and companion metadata.

---

## Dataset structure

```
huggingface_dataset/
 ├─ mega_licensed_corpus_redacted.txt      # Licensed Subset – concatenated code
 ├─ python_files.txt                       # Licensed Subset – raw file URLs
 ├─ python_files_elaborated.txt            # Elaborated Collection – raw file URLs
 ├─ python_files_elaborated_metadata.csv   # Elaborated Collection metadata
 └─ custom_tokens_vocab.txt             # `<token>\t<id>` vocabulary file
```

### File separator

Individual files are concatenated with the sentinel line:

```
# <FILESEP>
```

Anything following the sentinel until the next sentinel (or EOF) is the source
code of one file.

---

## Dataset variants

### 1. Licensed Subset  (`mega_licensed_corpus_redacted.txt`)

• 53 K permissively-licensed files (MIT/BSD/Apache/ISC/Unlicense).  
• All API keys & credentials removed.  
• Ready for redistribution & commercial use (respect upstream NOTICE files).

### 2. Elaborated Collection  (`python_files_elaborated.txt`)

• 186 K files from a much larger crawl.  
• Contains **GPL / LGPL / AGPL and other copyleft** licenses.  
• Shipped *as URL list* + metadata CSV; you must download the files yourself
  (`datasets.load_dataset` streaming, `wget`, etc.).  
• **No license filtering or secret-redaction performed** – use with caution.

When first loading the dataset, decide which variant aligns with your use case
(e.g. proprietary model training → Licensed Subset only).

---

## Collection methodology

1. **Repository discovery**

   - Queried GitHub REST API for projects with **≥ 10 stars**  
     (earlier iterations used 100+, later expanded for coverage).
   - Only repositories with primary language _Python_ and last commit ≥ 2015.

2. **File filtering**

   - Retain files whose **size ∈ [1 KB, 100 KB]**.
   - Exclude common build/packaging scripts (`setup.py`, `__init__.py`, etc.).

3. **License compliance**

   - Allowed: MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unlicense.
   - GPL, LGPL, AGPL and proprietary licenses were **excluded**.

4. **Deduplication**

   - Unique file SHA hashes; duplicates skipped.

5. **Formatting & cleaning**

   - Formatted with _autopep8_ to normalise whitespace.
   - Custom script removed trailing whitespace & normalised newlines.

6. **Secret redaction**
   - `truffleHog` + custom regex pass removed >150 active credentials.
   - Redacted corpus stored as `mega_licensed_corpus_redacted.txt`.

---

## Custom tokenisation

The accompanying `custom_tokens_vocab.txt` implements a **Python-aware
sub-token scheme**:

1. Strip doc-strings & comments.
2. Split on:
   - Camel-Case boundaries (`Camel` → `Camel`, `Case`)
   - Underscores, spaces
   - Indentation & newlines (preserved as `<newline>` token)
3. Rare tokens (frequency < 10) were dropped → 443 k vocabulary.

Example:

```python
def helloWorld(value):
    return value + 1
```

tokenises to:

```
def hello world ( value ) <newline>     return value + 1 <newline>
```

---

## Usage

```python
from datasets import load_dataset

ds = load_dataset("jblitzar/github-python", split="train")

print(ds[0]["code"][:300])        # raw source code
```

If you prefer token level examples (small reasons: memory), map the tokenizer:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("custom_tokens_vocab.txt")

def encode(ex):
    ex["input_ids"] = tok.encode(ex["code"]).ids
    return ex

ds = ds.map(encode, remove_columns=["code"])
```

---

## Ethical considerations & limitations

- **Licenses respected** – only permissive licenses included; retain NOTICE
  files when redistributing derivative works.
- **Secrets removed** – automated & manual audits performed, yet users **must
  not assume zero secrets**; re-audit before public deployments.
- **Code quality** – projects vary in style & correctness. Generated models
  may replicate bugs or vulnerable patterns.

---

## Citation

If you use this dataset, please cite:

```
@misc{github-python-2024,
  author       = {JBlitzar},
  title        = {GitHub-Python: A Permissively Licensed Corpus of Python Code},
  year         = {2024},
  howpublished = {\url{https://huggingface.co/datasets/jblitzar/github-python}},
  note         = {Version 1.0}
}
```

---

## License

Dataset card and aggregation scripts: **GPLv3**.  
Each code snippet remains under its **original repository license** (MIT,
Apache-2.0, BSD, ISC, etc.). Users must comply with upstream notices when
redistributing code or derivatives.
