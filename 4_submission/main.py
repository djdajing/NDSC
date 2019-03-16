import sys
import os
sys.path.append(os.path.abspath('../helpers'))
import Utilities as Utilities
import const
import pandas as pd



def format_csv_to_submit(df_input):
    df_output = pd.DataFrame(columns=const.SUBMISSION_COLUMNS_LIST)
    for index, row in df_input.iterrows():
        itemid = ""
        for col in df_input:
            if col == "itemid":
                itemid = str(int(row[col]))
            else:
                id=itemid+"_"+str(col)
                tag=str(row[col])
                temp = pd.DataFrame([[id,tag]],columns= ["id","tagging"])
                df_output=df_output.append(temp)
    return df_output



"""
TO RUN :

python main.py -d /home/dj/NDSC/test_csv/ -o /home/dj/NDSC/submission/data_info_val_sample_submission.csv -c all

"""
if __name__ == "__main__":
    args = Utilities.process_arg()
    csvs_folder_path= args["csvdir"]
    category = args["category"]
    out_dir = args["outputdataset"]

    if (category=="all"):
        categories = ['beauty','fashion','mobile']
    else :
        categories = [category]
    list_of_df =[]
    for category in categories :
        in_test_csv_path = csvs_folder_path + category + const.TEST
        print "category : ", category, in_test_csv_path

        #read csv as df
        df_ = pd.read_csv(in_test_csv_path, index_col=None) # read csv to be converted to submission format
        print "original df size ", df_.shape
        df = df_.drop(["title","image_path"],axis=1) # drop useless columns

        # write to csv for submission
        df = format_csv_to_submit(df)
        list_of_df.append(df)
        print "len df : ",df.shape
    df_to_save = pd.DataFrame(columns=const.SUBMISSION_COLUMNS_LIST)

    for i in range(0,len(list_of_df)):
        if i == 0:
            df_to_save = list_of_df[i]
        else:
            df_to_save=df_to_save.append(list_of_df[i])
    df_to_save.to_csv(out_dir, index=False)





