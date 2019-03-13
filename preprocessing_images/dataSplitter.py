import pandas as pd
import os, sys
import os
sys.path.append(os.path.abspath('../helpers'))

import shutil
from CsvHelper import CsvHelper
#constants
ITEM_ID = "itemid"
ITITLE = "title"
IMAGE_PATH ="image_path"



#python dataSplitter.py /home/dj/NDSC/data/test_in /home/dj/NDSC/data/test_out /home/dj/NDSC/csvs/mobile_data_info_val_competition.csv


def pathiterator(dir, out_dir):
    for file in os.listdir(dir):
        img_full_path = os.path.join(dir, file)
    # transform(img_full_path,out_dir)


def copy_image(file_name, in_dir_, out_dir_, success_count, failed_count):
    try:
        in_file_path = os.path.join(in_dir_, file_name)
        out_file_path = os.path.join(out_dir_, file_name)
        shutil.copy(in_file_path,out_file_path)
        success_count+=1;
    except IOError:
        failed_count+=1;
        print "Cannot copy file from "+ in_file_path+" to "+ out_file_path
        exit(1)
    return success_count,failed_count

def copy_images(column_series, in_dir, out_dir):
    success_count=0
    failed_count=0
    row_count = 0
    for row in column_series:
        row_count+=1
        file_name = row.split("/")[1]
        success_count,failed_count = copy_image(file_name, in_dir, out_dir, success_count, failed_count)

    print "\nSuccess count : " , success_count
    print "Failed count  : " ,failed_count
    print "Total count   :", success_count+failed_count," / ",row_count


def split(in_dir,out_dir,csv):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print "created folder :",out_dir
    csv_helper = CsvHelper()
    csv_helper.set_csv(csv)
    print "[INFO] using csv file : ",csv
    image_paths_df=csv_helper.get_single_column(IMAGE_PATH)
    copy_images(image_paths_df, in_dir, out_dir)
