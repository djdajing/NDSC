import sys
import os
sys.path.append(os.path.abspath('../helpers'))
import Utilities as Utilities
from CsvHelper import CsvHelper
import const
from DataObj import DataObj
import numpy as np
import pickle

#python main.py -i /home/dj/NDSC/new_data/beauty/beauty_image/ -o /home/dj/NDSC/models/ -c beauty -d /home/dj/NDSC/csvs/ -save true
#python main.py -i /home/dj/NDSC/models/ -c beauty -d /home/dj/NDSC/csvs/


def get_top_2_dict(classes, predicted_multiclass,test):
    preds_idx_ = np.argsort(predicted_multiclass, axis=1) #get position of lowest to higherst scroe
    preds_idx_top2 = preds_idx_[:,-2:] #get position of highest 2 score
    preds_idx = np.asarray([i[::-1]for i in preds_idx_top2]) #flip score to high to low

    pred_top1=[]
    d_predicted = {}
    for doc_index, doc in enumerate(test.data):
        d_predicted[doc]=[]
        for p in preds_idx[doc_index]:
            i_predicted_class = classes[p]
            i_predicted_prob = predicted_multiclass[doc_index][p]
            i_class_prob_pair = ((i_predicted_class, i_predicted_prob))
            d_predicted[doc].append(i_class_prob_pair)
        pred_top1.append(classes[preds_idx[doc_index][0]])
    pred_top1= np.asarray(pred_top1)

    for k, v in d_predicted.items():
        print k, v
    return pred_top1, d_predicted

if __name__ == "__main__":
    args = Utilities.process_arg()
    in_dir = args["inputdataset"]
    csvs_folder_path= args["csvdir"]
    category = args["category"]

    label = "Product_texture"
    model_path = Utilities.construct_filepath(in_dir,[category,label],".model")
    loaded_model = pickle.load(open(model_path,"rb"))

    #testing
    #pred_top1, pred_top2 = get_top_2_dict(text_clf_svm.classes_, predicted_multiclass,val.data)

