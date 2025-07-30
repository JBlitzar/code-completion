---
annotations_creators:
- author
language:
- python
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

# GitHub-Python

A **767 MB** corpus of permissively-licensed Python code drawn from public GitHub repositories.  
The dataset was created to support training and evaluation of **code-completion / generation** models.

## Dataset at a glance

|                       | Value |
|-----------------------|-------|
| Files                 | 53,017 `.py` files |
| Repositories          | 16,447 |
| Owners                | 12,515 |
| Compressed size       | 732 MB (`mega_licensed_corpus_redacted.txt`) |
| Vocabulary            | 443,431 tokens (`custom_tokens_vocab.txt`) |
| Time period           | Commits ≥ 2015-01-01 |
| License coverage      | MIT, Apache-2.0, BSD, ISC, Unlicense |
| Removed secrets       | ✅ – all hard-coded secrets/API keys redacted |

Numbers were obtained from the final redacted corpus and companion metadata.

---

## Dataset structure

```
huggingface_dataset/
 ├─ mega_licensed_corpus_redacted.txt   # concatenated code corpus
 ├─ python_files.txt                    # list of raw file URLs (1-per-line)
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

## Collection methodology

1. **Repository discovery**  
   - Queried GitHub REST API for projects with **≥ 10 stars**  
     (earlier iterations used 100+, later expanded for coverage).  
   - Only repositories with primary language *Python* and last commit ≥ 2015.

2. **File filtering**  
   - Retain files whose **size ∈ [1 KB, 100 KB]**.  
   - Exclude common build/packaging scripts (`setup.py`, `__init__.py`, etc.).

3. **License compliance**  
   - Allowed: MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unlicense.  
   - GPL, LGPL, AGPL and proprietary licenses were **excluded**.

4. **Deduplication**  
   - Unique file SHA hashes; duplicates skipped.

5. **Formatting & cleaning**  
   - Formatted with *autopep8* to normalise whitespace.  
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

* **Licenses respected** – only permissive licenses included; retain NOTICE
  files when redistributing derivative works.
* **Secrets removed** – automated & manual audits performed, yet users **must
  not assume zero secrets**; re-audit before public deployments.
* **Code quality** – projects vary in style & correctness. Generated models
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