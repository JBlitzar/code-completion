import os

with open(os.path.expanduser("~/torch_datasets/github-python/all_trains_subset_corpus/data/corpus_processed.txt"), "r") as f:
    data  = f.read()
    to_check = """ <newline> <newline> palette = palette _ demo ( ) <newline> <newline> net = caffe . segmenter ( prototxt , model , true"""
    #to_check = "<UNK>"
    to_check = to_check.replace(" ", "").lower()
    data = data.replace(" ", "").lower()
    
    print(to_check in data)