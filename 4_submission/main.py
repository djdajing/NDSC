import sys
import os
sys.path.append(os.path.abspath('../helpers'))
import Utilities as Utilities
import const
import pandas as pd
"""
TO RUN :

python main.py -d /home/dj/NDSC/csvs/ -o /home/dj/NDSC/submission/ -c beauty

"""


def format_csv_to_submit(df_input, out_path):
    df_output = pd.DataFrame(columns=const.SUBMISSION_COLUMNS_LIST)
    for index, row in df_input.head().iterrows():
        itemid = ""
        for col in df_input:
            if col == "itemid":
                itemid = str(row[col])
            else:
                id=itemid+"_"+str(col)
                tag=str(row[col])
                temp = pd.DataFrame([[id,tag]],columns= ["id","tagging"])
                df_output=df_output.append(temp)
    df_output.to_csv(out_path, index=False)


if __name__ == "__main__":
    args = Utilities.process_arg()
    csvs_folder_path= args["csvdir"]
    category = args["category"]
    out_dir = args["outputdataset"]

    in_test_csv_path = csvs_folder_path + category + const.TEST
    out_test_csv_path = out_dir + category + const.TEST

    #read csv as df
    df_ = pd.read_csv(in_test_csv_path, index_col=None) # read csv to be converted to submission format
    df = df_.drop(["title","image_path"],axis=1) # drop useless columns

    # write to csv for submission
    format_csv_to_submit(df,out_test_csv_path)





