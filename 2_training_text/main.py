from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
#%matplotlib inline
import sys
import os
sys.path.append(os.path.abspath('../helpers'))
import Utilities
from CsvHelper import CsvHelper
import const
from DataObj import DataObj
import numpy as np


#Training imports
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
import pickle

def make_obj_dataobj(X, y):
    Obj= DataObj()
    Obj.set_data(X)
    Obj.set_target(y)
    return Obj


def matrix_report_void(ground_truth, target_predicted_top1, targets_str, targets_float):
    print('accuracy %s' % accuracy_score(target_predicted_top1, ground_truth))
    print(classification_report(ground_truth, target_predicted_top1, target_names=targets_str, labels=targets_float))


def get_top_1_arr(classes,predicted_multiclass,val_data):
    preds_idx_ = np.argsort(predicted_multiclass, axis=1) #get position of lowest to higherst scroe
    preds_idx_top2 = preds_idx_[:,-2:] #get position of highest 2 score
    preds_idx = np.asarray([i[::-1]for i in preds_idx_top2]) #flip score to high to low

    pred_top1=[]

    for i, d in enumerate(val_data):
        top1 =classes[preds_idx[i][0]]
        print d, "=>" ,top1
        pred_top1.append(top1)
    pred_top1= np.asarray(pred_top1)
    return pred_top1




if __name__ == "__main__":
    save_model =False
    args = Utilities.process_arg("-save")
    in_dir = args["inputdataset"]
    out_dir = args["outputdataset"]
    csvs_folder_path= args["csvdir"]
    category = args["category"]
    save =args["save"]
    if (save == "t" or save == "T" or save == "true"):
        save_model=True

    train_csv_path = csvs_folder_path + category + const.TRAIN

    print train_csv_path
    csv_helper = CsvHelper()
    csv_helper.set_csv(train_csv_path)

    label= "Product_texture"

    df = csv_helper.get_id_title_and_single_column(label)
    X = df["title"]
    y = df[label]

    #get all
    y_unqiue = y.unique()
    targets_str = [str(i) for i in y_unqiue]
    targets_float = [float(i) for i in y_unqiue]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
    train = make_obj_dataobj(X_train, y_train)
    val = make_obj_dataobj(X_val, y_val)

    #training
    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5,
                                                       random_state=42))])
    text_clf_svm = text_clf_svm.fit(train.data,train.target)

    predicted_multiclass = text_clf_svm.predict_proba(val.data)

    pred_top1 = get_top_1_arr(text_clf_svm.classes_, predicted_multiclass,val.data)

    matrix_report_void(val.target, pred_top1, targets_str, targets_float)

    if save_model:
       # saving_fileName =category+"_"+label+".model"
        saving_path =Utilities.construct_filepath(out_dir, [category,label], ".model")
        pickle.dump(text_clf_svm,open(saving_path,'wb'))


