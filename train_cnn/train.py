
import os
import cv2

import sys
import os
sys.path.append(os.path.abspath('../helpers'))

import sys
from CsvHelper import CsvHelper
import numpy as np
# import the necessary packages
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


def select_dataset(choice):
    chosen_dataset = "beauty_image"
    if choice == "2":
        chosen_dataset = "fashion_image"
    elif choice == "3":
        chosen_dataset = "mobile_image"
    print "DATASET SELECTED : ", chosen_dataset
    return chosen_dataset


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


def get_label_headers(csv_helper,csv_path,img_set):
    csv_helper.set_csv(csv_path)
    csv_helper.set_category(img_set)
    csv_helper.set_all_headers_as_label()
    #csv_helper.set_label_headers(4, -1)
    return csv_helper.get_label_headers()


def binarizer(labels_to_use):
    """"returns a dictionary where key is the label title and value is tuple of (nparray of key in binary fform, number of classes)"""
    binarized_labels= {}
    for label in labels_to_use:
        binarizer  = LabelBinarizer()
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
        ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
                   label="val_" + l)
        ax[i].legend()

    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plt.savefig("{}_losses.png".format(out_dir))
    plt.close()

if __name__ == "__main__":

    image_dir = sys.argv[1]
    csv_path = sys.argv[2]
    out_dir = sys.argv[3]
    dataset_choice = sys.argv[4]

    img_set = select_dataset(dataset_choice)

    csv_helper = CsvHelper()
    label_header = get_label_headers(csv_helper,csv_path,img_set)
    data, all_labels = set_data_labels(image_dir, csv_helper)

    wanted_categories = ["Brand","Color Family"]
    wanted_categories_loss = ["brand_loss","colour_family_loss"]
    labels_to_use= get_labels_to_use(label_header, all_labels, wanted_categories)

    binarized_labels = binarizer(labels_to_use)

    screenSizeLabels = binarized_labels["Brand"][0]
    screenSizeLabels_len = binarized_labels["Brand"][1]
    colorLabels = binarized_labels["Color Family"][0]
    colorLabels_len = binarized_labels["Color Family"][1]

    #train
    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    split = train_test_split(data, screenSizeLabels, colorLabels,
                             test_size=0.2, random_state=42)
    (trainX, testX, trainScreenSizeY, testScreenSizeY,
     trainColorY, testColorY) = split


    model= MobileNet.build(IMG_DIM,IMG_DIM,numCategories=screenSizeLabels_len,numColors=colorLabels_len,finalAct="softmax")

    # define two dictionaries: one that specifies the loss method for
    # each output of the network along with a second dictionary that
    # specifies the weight per loss
    losses = {"screen_size_output": "categorical_crossentropy",
        "color_output": "categorical_crossentropy",
        }
    lossWeights = {"screen_size_output": 1.0, "color_output": 1.0}

    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
                  metrics=["accuracy"])

    # train the network to perform multi-output classification
    H = model.fit(trainX,
                  {"screen_size_output": trainScreenSizeY, "color_output": trainColorY},
                  validation_data=(testX,
                                   {"screen_size_output": testScreenSizeY, "color_output": testColorY}),
                  epochs=EPOCHS,
                  verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    print out_dir,".model"
    model.save(out_dir+".model")

    wanted_categories = ["screen_size_output_loss","color_output_loss"]
    plot_loss(H,wanted_categories)