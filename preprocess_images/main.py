import sys
import argparse
import os
import resizeImg
import dataSplitter

#PATH_RESIZED_IMG =
TRAIN = "_data_info_train_competition.csv"
TEST = "_data_info_val_competition.csv"

def check_path(path):
    if not os.path.exists(path):
        raise IOError("Path Does Not Exitst : "+path)
    else:
        return


def process_arg():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputdataset", required=True,help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-o", "--outputdataset", required=True,help="path to outputdataset")
    ap.add_argument("-c", "--category", required=True,help="state category")
    ap.add_argument("-d", "--csvdir", required=True,help="state csvdir")

    return vars(ap.parse_args())

if __name__ == "__main__":


    args = process_arg()

    in_dir = args["inputdataset"]
    out_dir = args["outputdataset"]
    csvs_folder_path= args["csvdir"]
    category = args["category"]

    check_path(in_dir)
    check_path(out_dir)
    check_path(csvs_folder_path)

    train_csv_path = csvs_folder_path + category + TRAIN
    test_csv_path = csvs_folder_path + category + TEST

    #do resizing
    resize_output_folder_name = category + "_image_128"
    resize_output_folder_path = out_dir + resize_output_folder_name
    resizeImg.resize(in_dir, resize_output_folder_path)

    #do data splitting
    train_output_folder_name = category + "_image_train"
    test_output_folder_name = category + "_image_test"
    split_train_output_folder_path = out_dir + train_output_folder_name
    split_test_output_folder_path = out_dir + test_output_folder_name


    dataSplitter.split(resize_output_folder_path, split_test_output_folder_path, test_csv_path)
    dataSplitter.split(resize_output_folder_path, split_train_output_folder_path, train_csv_path)