# Code Completion

# Take a look at [notes.md](https://github.com/JBlitzar/code-completion/blob/main/NOTES.md)

## **1. Overview**

This project aims to develop a code completion model for Python. My process involves scraping GitHub repositories, preprocessing the data, implementing and training a transformer, and refining its performance to improve generalization and avoid overfitting.

## **2. Data Collection & Processing**

- See [dataset.py](dataset.py)
- Scraped GitHub repositories with the following filters ([link](scraping)):
  - More than 100 stars
  - After 2015 (to ensure usage of modern python)
  - `.py` files between 1KB and 100KB
- Processed around 30,000 repositories and filtered them based on SHA hashes to avoid duplicates, resulting in about 500MB of text data and 53,000 files.
- Files are then formatted using `autopep8`
- Tokenization experiments:
  - Started with BERT-based tokenization.
  - Explored Byte Pair Encoding (BPE) with `yttm` but decided against it because of strange tokenization issues. Processing code is a more nuanced problem than natural language (different use of punctuation and whitespace, in particular).
  - Eventually settled on a custom tokenizer ([link](https://github.com/JBlitzar/code-completion/blob/main/dataset.py#L178)), agressively subdividing the code by first removing docstrings and comments, and then splitting based off of capitalization, spaces, and underscores while preserving newlines and indentation.
    - I discovered that, despite the agressive tokenization, there were still many tokens that were used only once or twice. In the end, I only preserved tokens that appeared more than ten times.

## **3. Model Development & Training**

- After learning about attention mechanisms and reading through various [resources](resources.md), I implemented it myself in [architecture.py](architecture.py). The design is very modular, each component usually being composed of a few smaller components glued together in a `Sequential`. While this was an excellent learning opportunity, and it was really great to truly understand how attention mechanisms worked inside of a transformer, because this project has so many moving parts, as I continued debugging, I used pytorch's builtin implementation of transformers for iteration. The [source code itself](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L57) is actually surprisingly similar.
- I created my own [training framework](trainingmanager.py), which I've used in the past to quickly train other models. Building off of this, I made a quick script to run [hyperparameter searches](hyperparam_tune.py).
- I implemented gradient clipping, weight decay, and Xavier normalization.
- What's amazing is that I was at this stage of the project _in November_. In previous, less complex ML projects (such as the VAE), I would do a few weeks of training and finetuning, but usually finish not long after that.

## **4. Challenges, Takeaways, & Next Steps**

### **Challenges**

- Many challenges arose while training.
- First of all, I was getting NaNs in the loss, due to incorrect casting in the Multi-Head Attention. At this point, I decided to use the builtin implementation in order to isolate the problem and prevent future issues like this.
- This next but is probably the most intense one I faced during this project. An interesting shape issue arose where the model expected data in the shape of (seq_len, batch_size), but was receiving and outputting the reverse. What was insane was that in the loss calculation, I flattened the outputs, leading to a lack of actual errors. Pytorch is usually good at catching shape errors and making debugging easy, but if you transpose and then flatten, it would have the same shape as if you didn't.

  - I only discovered this after actual weeks of debugging and scrutinization of other parts of the code. Finally, I was able to isolate it to this after training on a [purposely undersized dataset](dummy-data-dir/data/corpus.txt) to get the model to overfit, and noticing an incorrect pattern in the outputs.
  - While this fixed the core issue, the bug persisted in a couple of ways:
    - I previously had two places where I evaluated the model in `trainingmanager.py`: The training step and the validation step. I didn't fix the issue in the validation step, which caused it to persist and validation loss to increase rather than decrease over time, creating misleading graphs that looked like overfitting.
    - I also saved the best checkpoint based off of lowest validation loss. Then, when [evaluating the model](eval.py), I loaded in the best checkpoint. Unfortunately, this lead to loading in of bad models because the validation loss was messed up.
    - The lesson here is to make sure you don't have the same code in multiple places, and to ensure when you change a part, that it won't have unintended side effects.

- Analysis of data is important to understand it and know how to treat it.
  - For example, it was important to realize that many tokens were used only a couple of times. This allowed me to cut down on the number of unique tokens, thus reducing model size, without disrupting the diversity of the dataset.

### **Next Steps**

- After relatively extensive hyperparameter tuning, I had determined that I was plateuing on performance.
- I am currently working on investigating new evaluation metrics, inference strategies such as beam search, and curating a larger dataset (with 10 github stars rather than 100 github stars as the minimum threshold, resulting in ~4x the data).

## **5. Conclusion**

This project has come a long way, from early scraping experiments to training a functioning code completion model. The focus now is on scaling up data collection and refining the model to produce high-quality completions that generalize well across unseen codebases.
