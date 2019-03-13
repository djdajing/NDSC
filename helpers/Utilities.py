import argparse
import os


def process_arg():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputdataset", required=True,help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-o", "--outputdataset", required=True,help="path to outputdataset")
    ap.add_argument("-c", "--category", required=True,help="state category")
    ap.add_argument("-d", "--csvdir", required=True,help="state csvdir")

    return vars(ap.parse_args())

def check_path(path):
    if not os.path.exists(path):
        raise IOError("Path Does Not Exitst : "+path)
    else:
        return