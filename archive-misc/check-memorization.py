import os

with open(
    os.path.expanduser(
        "~/torch_datasets/github-python/mega_corpus/data/corpus_processed.txt"
    ),
    "r",
) as f:
    data = f.read()
    to_check = """<newline> logger . info ( f " initial validation samples in first step . . . " ) <newline> model . eval ( ) <newline> <newline> gen _ validation _ samples ( validation _ pipeline , args , wandb , samples _ dir , train _ ts , train _ steps ) <newline> <newline> model . train ( ) <newline>"""
    to_check = """' nonpayable ' , ' type ' : ' function ' } , { ' inputs ' : [ { ' internaltype ' : ' uint 2 5 6 ' , ' name ' : ' ' , ' type ' : ' uint 2 5 6 ' } ] , ' name ' : ' ' , ' outputs"""

    to_check = """parser . add _ argument ( ' - - save _ folder ' , type = str , default = ' data / save ' , help = ' save folder ' )"""
    to_check = """= torch . zeros ( len ( imgs ) ) <newline> <tab> for x _ interp in range ( 1 , args . batch _ size ) :"""
    # to_check = """x _ interp = machine . interpolate ( imgs [ 0 ] , imgs [ 1 ] , n _ interp )""" # should be true
    # to_check = "<UNK>"
    to_check = to_check.replace(" ", "").lower()
    data = data.replace(" ", "").lower()

    print(to_check in data)
