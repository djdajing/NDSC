import argparse
import os


def process_arg(addarg=None):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputdataset", required=False,help="input directory (could be image or model)")
    ap.add_argument("-o", "--outputdataset", required=False,help="output directory")
    ap.add_argument("-c", "--category", required=False,help="category")
    ap.add_argument("-d", "--csvdir", required=False,help="directory where csv files are")
    if addarg:
        ap.add_argument(addarg,required=False)


    return vars(ap.parse_args())

def check_path(path):
    if not os.path.exists(path):
        raise IOError("Path Does Not Exitst : "+path)
    else:
        return

def construct_filepath(path, names,ext):
    file_path =path
    for i in range( 0, len(names)):
        add = names[i]
        if (i!=len(names)-1):
            add +="_"
        file_path+=add
        i+=1
    file_path+=ext
    return file_path