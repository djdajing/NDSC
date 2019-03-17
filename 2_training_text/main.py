from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV,StratifiedKFold

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
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
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
    print('f1 %s' % f1_score(target_predicted_top1, ground_truth,average='micro'))
    print('precision %s' % precision_score(target_predicted_top1, ground_truth,average='micro'))
    print('recall %s' % recall_score(target_predicted_top1, ground_truth,average='micro'))

    #print(classification_report(ground_truth, target_predicted_top1, target_names=targets_str, labels=targets_float))


def split_data_stratified(X, y):
    #split by proportion
    sssplit = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=1)
    for train_index , test_index in sssplit.split(X,y):
        X_train, X_val = X[train_index],X[test_index]
        y_train,y_val= y[train_index],y[test_index]
    return  X_train, X_val, y_train, y_val


def get_top_1_arr(classes,predicted_multiclass,val_data):
    preds_idx_ = np.argsort(predicted_multiclass, axis=1) #get position of lowest to higherst scroe
    preds_idx_top2 = preds_idx_[:,-2:] #get position of highest 2 score
    preds_idx = np.asarray([i[::-1]for i in preds_idx_top2]) #flip score to high to low

    pred_top1=[]

    for i, d in enumerate(val_data):
        top1 =classes[preds_idx[i][0]]
        pred_top1.append(top1)
    pred_top1= np.asarray(pred_top1)
    return pred_top1


def _train(X,y):
    y_unique = y.unique()
    print "unique : ", y_unique
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    from imblearn.over_sampling import SMOTE
    # Pipeline item
    cv = CountVectorizer()  # make into bag of words
    tfidf = TfidfTransformer()  # apply tfidf
    upsampling =  SMOTE(k_neighbors=1)
    svm = SGDClassifier(penalty='l2', loss='modified_huber')

    parameters = {
        'cv__stop_words': ('english', None),
        'cv__max_df': (0.8, 0.9, 0.85),
        'tfidf__use_idf': (True, False),
        'svm__alpha': (1e-3, 1e-4),
        'svm__max_iter': (5000, 10000),
        'svm__tol': (1e-4, 1e-3)
    }

    # training
    text_clf_svm = Pipeline([('cv', cv), ('tfidf', tfidf), ('upsampling',upsampling),('svm', svm)])
    gs_clf = GridSearchCV(text_clf_svm, parameters, n_jobs=-1, cv=skf.split(X, y), scoring='f1_micro')

    gs_clf = gs_clf.fit(X, y)

    print "Best Parameter : ", gs_clf.best_params_
    print "F1 Score : ", gs_clf.best_score_

    print "============================================"

    # Saving model
    if save_model:
        saving_path = Utilities.construct_filepath(out_dir, [category, label], ".model")
        # pickle.dump(text_clf_svm,open(saving_path,'wb'))


def _proprocess(df,label,X,y):

    target_cout = df[label].value_counts().to_frame()
    #print target_cout
    from imblearn.over_sampling import RandomOverSampler
    smote = RandomOverSampler(random_state=15)
    X_,y_ = smote.fit_sample(X,y)
    print type(X),type(y)

def do_training(label,csv_helper,save_model):

    print "========  TRAINING MODEL : ",label,"========"

    df = csv_helper.get_id_title_and_single_column(label) # get id, title, and label column

    X = df["title"]
    y = df[label]
    #_proprocess(df, label,X,y)

    _train(X,y)



# def do_training(label,csv_helper,save_model):
#
#     print "TRAINING MODEL : ",label
#
#     df = csv_helper.get_id_title_and_single_column(label) # get id, title, and label column
#
#     X = df["title"]
#     y = df[label]
#
#     # pre-training
#     y_unqiue = y.unique() # get a list of labels
#     targets_str = [str(i) for i in y_unqiue] #string verison of y_unique
#     targets_float = [float(i) for i in y_unqiue] # float version of y-unqiue
#
#     #X_train, X_val, y_train, y_val  = split_data_stratified(X.values, y.values)
#
#     #print "Train sample:", X_train.shape,y_train.shape
#     #print "Test sample:", X_val.shape,y_val.shape
#
#     #train = make_obj_dataobj(X_train, y_train)
#     #val = make_obj_dataobj(X_val, y_val)
#
#     skf=StratifiedKFold(n_splits= 10,shuffle = True,random_state=42)
#
#     # Pipeline item
#     cv = CountVectorizer() # make into bag of words
#     tfidf= TfidfTransformer()
#     svm =SGDClassifier(penalty='l2',loss='modified_huber')
#
#     parameters ={
#         'cv__stop_words':('english',None),
#         'cv__max_df':(0.8,0.9,0.85),
#         'tfidf__use_idf':(True, False),
#         'svm__alpha':(1e-3,1e-4),
#         'svm__max_iter':(5000,10000),
#         'svm__tol':(1e-4,1e-3)
#     }
#
#     # training
#     text_clf_svm = Pipeline([('cv', cv), ('tfidf', tfidf), ('svm', svm)])
#     gs_clf = GridSearchCV(text_clf_svm,parameters,n_jobs=-1,cv=skf.split(X,y),scoring='f1_micro')
#
#     gs_clf = gs_clf.fit(X,y)
#
#     print gs_clf.best_score_
#
#     #text_clf_svm=text_clf_svm.set_params(**gs_clf.best_params_)
#
#     # make prediction
#     #predicted_multiclass = gs_clf.best_estimator_.predict_proba(val.data)
#     #pred_top1 = get_top_1_arr(gs_clf.classes_, predicted_multiclass,val.data) #get top 1 for evaluation
#     #matrix_report_void(val.target, pred_top1, targets_str, targets_float) #evaluation
#
#     # Saving model
#     if save_model:
#         saving_path =Utilities.construct_filepath(out_dir, [category,label], ".model")
#         #pickle.dump(text_clf_svm,open(saving_path,'wb'))

"""
TO RUN :

python main.py  -o /home/dj/NDSC/models/ -c beauty -d /home/dj/NDSC/csvs/ -save true

if arguement -save true, it will save model into -o folder with format refer to "Utilites" "construct_filepath" function 

"""

if __name__ == "__main__":

    save_model =False
    args = Utilities.process_arg("-save")
    out_dir = args["outputdataset"]
    csvs_folder_path = args["csvdir"]
    category = args["category"]
    save = args["save"]

    if save == "t" or save == "T" or save == "true":
        save_model = True

    train_csv_path = csvs_folder_path + category + const.TRAIN

    csv_helper = CsvHelper()
    csv_helper.set_csv(train_csv_path) # set csv file as train_csv_path

    '''
    This part is for mass training like for each attribute/ label just train a model 
    for individual model tuning, set labels with a single label , like like 123
    '''
    csv_helper.set_all_headers_as_label() # set all the labels as label headers
    labels = csv_helper.get_label_headers() # get lables i.e all column name besides image name, itemid and title
    labels = ["Camera"] # just doing
    for label in labels:
        print label
        do_training(label,csv_helper,save_model)


