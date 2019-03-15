import sys
import os
sys.path.append(os.path.abspath('../helpers'))
import Utilities as Utilities
from CsvHelper import CsvHelper
import const
from DataObj import DataObj
import numpy as np
import pickle


def get_top_2_dict(classes, predicted_multiclass,id_df):
    preds_idx_ = np.argsort(predicted_multiclass, axis=1) #get position of lowest to higherst scroe
    preds_idx_top2 = preds_idx_[:,-2:] #get position of highest 2 score
    preds_idx = np.asarray([i[::-1]for i in preds_idx_top2]) #flip score to high to low

    top2_dict = {}
    for doc_index, id in enumerate(id_df):
        top2_dict[id]=[] #id is the key
        for p in preds_idx[doc_index]:
            i_predicted_class = classes[p] # class predicted
            i_predicted_prob = predicted_multiclass[doc_index][p] # confidence
            i_class_prob_pair = ((i_predicted_class,i_predicted_prob)) # value = (class, prob)
            top2_dict[id].append(i_class_prob_pair) # e.g. 661957090 [(8.0, 1.0), (7.0, 0.0)]
    return top2_dict


def get_top_classes(top2_list):
    predicted_classes_str =""
    for pair in top2_list:
        predicted_class = pair[0]
        predicted_confidence = pair[1]
        if predicted_confidence >= const.CONFIDENCE:
            predicted_classes_str= predicted_classes_str+str(int(predicted_class))+" "
    return predicted_classes_str


def write_to_csv(pred_top2,test_df):
    # Write to csv
    for id, top2_list in pred_top2.items():
        target_str = get_top_classes(top2_list) # function to decide what class to keep
        idx = test_df.index[test_df["itemid"]==id] # get index of id
        test_df.loc[idx,label]= target_str # use id to find the correct row and insert string
    test_df.to_csv(test_csv_path,index=False)

"""
TO RUN :

python main.py -i /home/dj/NDSC/models/ -c beauty -d /home/dj/NDSC/csvs/

"""

if __name__ == "__main__":

    args = Utilities.process_arg()
    in_dir = args["inputdataset"]
    csvs_folder_path= args["csvdir"]
    category = args["category"]

    # get model for "label"
    label = "Brand"
    model_path = Utilities.construct_filepath(in_dir,[category,label],".model")
    text_clf_svm = pickle.load(open(model_path,"rb"))

    # read in test csv data
    test_csv_path = csvs_folder_path + category + const.TEST
    csv_helper = CsvHelper()
    csv_helper.set_csv(test_csv_path)
    csv_helper.add_column(label) # create new column with title label (set on top) so that we can fil lin later
    test_df = csv_helper.get_csv()

    # Do prediction
    data_df = test_df["title"]
    id_df = test_df["itemid"]
    predicted_multiclass = text_clf_svm.predict_proba(data_df)
    pred_top2 = get_top_2_dict(text_clf_svm.classes_, predicted_multiclass,id_df)

    # write back to csv
    write_to_csv(pred_top2, test_df) # top 2 is the top 2 predicted test_df is the original csv




