import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
#%matplotlib inline
import sys
import os
sys.path.append(os.path.abspath('../helpers'))
import Utilities
from CsvHelper import CsvHelper


#Training imports
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

TRAIN = "_data_info_train_competition.csv"


if __name__ == "__main__":
    args = Utilities.process_arg()
    in_dir = args["inputdataset"]
    out_dir = args["outputdataset"]
    csvs_folder_path= args["csvdir"]
    category = args["category"]

    train_csv_path = csvs_folder_path + category + TRAIN

    csv_helper = CsvHelper()
    csv_helper.set_csv(train_csv_path)
    df = csv_helper.get_title_and_single_column("Product_texture")
    X = df["title"]
    y = df["Product_texture"]

    y_unqiue = y.unique()
    my_tags = [str(i) for i in y_unqiue]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print len(X_train),len(y_train)
    print len(X_test),len(y_test)

    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)

    #%%time

    y_pred = sgd.predict(X_test)
    print y_pred

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred, target_names=my_tags))