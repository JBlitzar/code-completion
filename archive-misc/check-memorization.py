import os

with open(os.path.expanduser("~/torch_datasets/github-python/corpus/data/corpus_processed.txt"), "r") as f:
    data  = f.read()
    to_check = " manager ( ) <newline> pmb . populate ( pm ) <newline> pm . run ( llvmmod ) <newline> <newline> if llvmdump : <newline> print ( ' = = = = = = = = optimized llvm ir ' ) <newline> print ( str ( llvmmod ) ) <newline>"
    #to_check = "<UNK>"
    to_check = to_check.replace(" ", "").lower()
    data = data.replace(" ", "").lower()
    
    print(to_check in data)