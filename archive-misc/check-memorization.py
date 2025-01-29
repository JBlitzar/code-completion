import os

with open(os.path.expanduser("~/torch_datasets/github-python/all_trains_subset_corpus/data/corpus_processed.txt"), "r") as f:
    data  = f.read()
    to_check = """<newline> logger . info ( f " initial validation samples in first step . . . " ) <newline> model . eval ( ) <newline> <newline> gen _ validation _ samples ( validation _ pipeline , args , wandb , samples _ dir , train _ ts , train _ steps ) <newline> <newline> model . train ( ) <newline>"""
    #to_check = "<UNK>"
    to_check = to_check.replace(" ", "").lower()
    data = data.replace(" ", "").lower()
    
    print(to_check in data)