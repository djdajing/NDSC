import sys
import os
sys.path.append(os.path.abspath('../helpers'))

import sys
from CsvHelper import CsvHelper
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from MobileNet import MobileNet
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

#constants
EPOCHS= 3
LR = 1e-3
BATCH_SIZE=32
IMG_DIMENSION = (128,128,3)
IMG_DIM = 128

def process_arg():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputdataset", required=True,help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-o", "--outputdataset", required=True,help="path to outputdataset")
    ap.add_argument("-c", "--category", required=True,help="state category")
    ap.add_argument("-d", "--csvdir", required=True,help="state csvdir")

    args = vars(ap.parse_args())

    in_dir = args["inputdataset"]
    out_dir = args["outputdataset"]
    category = args["category"]
    csv_path= args["csvdir"]+ category+"_data_info_train_competition.csv"

    return in_dir, out_dir,category,csv_path


def set_data_labels(image_dir, csv_helper):
    data_ = []
    labels_ =[]
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir,img_name)
        img = cv2.imread(img_path)
        img = img_to_array(img)
        lables_of_single_img = csv_helper.get_label_by_img_name(img_name)
        labels_.append(lables_of_single_img)
        data_.append(img)

    labels_ = np.array(labels_)
    data_ = np.array(data_)

    return data_, labels_


def data_normalisation(data):
    return np.array(data,dtype="float")/255.0


def data_dimension_checking(data, training_data, label):
    title = " Testing "
    type = " input "
    if training_data :
        title = " Training "
    if label:
        type = " label "
    print len(data) ,title,type


def get_label_headers(csv_helper,csv_path,category):
    csv_helper.set_csv(csv_path)
    csv_helper.set_category(category)
    csv_helper.set_all_headers_as_label()
    #csv_helper.set_label_headers(4, -1)
    return csv_helper.get_label_headers()


def binarizer(labels_to_use):
    """"returns a dictionary where key is the label title and value is tuple of (nparray of key in binary fform, number of classes)"""
    binarized_labels= {}
    for label in labels_to_use:
        binarizer = LabelBinarizer()
        label_i = binarizer.fit_transform(labels_to_use[label])
        binarized_labels[label]=((label_i,len(binarizer.classes_)))

    for key in binarized_labels:
        print "[INFO] Binarised subategory: ",key,"with", len(binarized_labels[key][0]),"samples &",binarized_labels[key][1],"classes"
    return binarized_labels


def get_labels_to_use(label_header,all_labels,wanted_labels):
    label_header_count = len(label_header)
    labels = {}

    for wanted_label in wanted_labels:
        for i in range (0,label_header_count):
            if wanted_label == label_header[i]:
                labels[wanted_label] = list(all_labels[:, i])

    # for key in labels:
    #     print "test",key, len(labels[key])
    return labels


def plot_loss(H, losses):
    # plot the total loss, category loss, and color loss
    lossNames = ["loss"] + losses
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

    # loop over the loss names
    for (i, l) in enumerate(lossNames):
        print "i :",i
        print "l :", l
        # plot the loss for both the training and validation data
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
        ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],label="val_" + l)
        ax[i].legend()

    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plt.savefig("{}_losses.png".format(out_dir))
    plt.close()


def gen_name(_list ,to_add):
    new_list =[]
    for item in _list:
        new_item = item.lower().replace(" ", "_")+to_add
        new_list.append(new_item)
    return new_list


# e.g. {"Brain ",(array_of_labels, number of classes)
def get_binarised_lable(_wanted_categories,index):
    return binarized_labels_dict[_wanted_categories[index]][0]


# e.g. {"Brain ",(array_of_labels, number of classes)
def get_num_of_classes(_wanted_categories,index):
    return binarized_labels_dict[_wanted_categories[index]][1]


def category_classcount_match(wanted_categories):
    _cat_classcount_match_dict ={}
    index = 0
    for item in wanted_categories:
        _cat_classcount_match_dict[item] = get_num_of_classes(wanted_categories,1)
        index+=1
    return  _cat_classcount_match_dict

if __name__ == "__main__":
    image_dir, out_dir, category, csv_path =process_arg()
    csv_helper = CsvHelper()
    label_header_list = get_label_headers(csv_helper, csv_path, category + "_image")
    data, all_labels = set_data_labels(image_dir, csv_helper)

    '''NEED MANUAL '''

    wanted_categories = ["Brand","Color Family"]
    wanted_categories_lower = gen_name(wanted_categories,to_add="")
    wanted_categories_output = gen_name(wanted_categories,to_add="_output")
    wanted_categories_loss = gen_name(wanted_categories,to_add="_output_loss")

    # selects the labels listed in wanted categories
    labels_to_use= get_labels_to_use(label_header_list, all_labels, wanted_categories)

    # binarize the labels
    binarized_labels_dict = binarizer(labels_to_use) #e.g. {"Brain ",(array_of_labels, number of classes)}


    # get labels and number of classes in the label
    brandLabels = get_binarised_lable(wanted_categories, 0)
    #brandLabels_len = get_num_of_classes(wanted_categories, 0)
    colorLabels = get_binarised_lable(wanted_categories,1)
    #colorLabels_len = get_num_of_classes(wanted_categories,1)

    cat_classcount_match_dict = category_classcount_match(wanted_categories)


    #train , 20% for testing,random_state is the seed used by the random number generator;
    split = train_test_split(data, brandLabels, colorLabels, test_size=0.2, random_state=42)
    (trainX, testX, trainScreenSizeY, testScreenSizeY,trainColorY, testColorY) = split


    model= MobileNet.build(IMG_DIM, IMG_DIM, catclasscountmatch=cat_classcount_match_dict, finalAct="softmax")

    # define two dictionaries: one that specifies the loss method for each output of the network

    losses={
            wanted_categories_output[0]: "categorical_crossentropy",
            wanted_categories_output[1]: "categorical_crossentropy",
            }
    # specifies the weight per loss
    lossWeights = {wanted_categories_output[0]: 1.0, wanted_categories_output[0]: 1.0}

    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,metrics=["accuracy"])

    # train the network to perform multi-output classification
    H = model.fit(trainX,
                  {
                      wanted_categories_output[0]: trainScreenSizeY,
                      wanted_categories_output[1]: trainColorY
                  },
                  validation_data=(testX,
                                   {
                                       wanted_categories_output[0]: testScreenSizeY,
                                       wanted_categories_output[1]:testColorY
                                   }),
                  epochs=EPOCHS,
                  verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    print out_dir,".model"
    model.save(out_dir+".model")
    plot_loss(H,wanted_categories_loss)