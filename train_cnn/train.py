
import os
import cv2
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


def set_data_labels(image_dir, csv_helper):
    data = []
    labels =[]
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir,img_name)
        img = cv2.imread(img_path)
        img = img_to_array(img)
        lables_of_single_img = get_label(img_name,csv_helper)

        labels.append(lables_of_single_img)
        data.append(img)
    labels = np.array(labels)
    #print labels
    return data,labels

def get_label(img_name, csv_helper):
    labels = csv_helper.get_label_by_img_name(img_name)
    return labels

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

    #image_paths_df = csv_helper.get_multi_column(IMAGE_PATH)
# def get_arg():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i","--inputdir",required=True,help="path to input dataset")
#     ap.add_argument("-m","--model",required=True,help="path to output model")
#     ap.add_argument("-l","--labelbin",required=True,help="path to output label binarizer")
#     ap.add_argument("-p", "--plot", required=True, help="path to output label accuracy/lost plot")
#     args = vars(ap.parse_args())
if __name__ == "__main__":
    image_dir = sys.argv[1]
    csv_path = sys.argv[2]
    out_dir = sys.argv[3]

    csv_helper = CsvHelper()
    csv_helper.set_csv(csv_path)
    csv_helper.set_category("mobile_image")
    csv_helper.set_label_headers(4, -1)
    label_header = csv_helper.get_label_headers()

    data, all_labels = set_data_labels(image_dir, csv_helper)

    # creates a list of np array of size number_of_label * number_of_data_entry
    label_header_count = len(label_header)
    labels ={}
    for i in range (0,label_header_count):
        labels[label_header[i]] = list(all_labels[:, i])

    #do colour
    color_label = np.array(labels["Color Family"])
    screen_size_label = np.array(labels["Brand"])

    data = np.array(data)

    # binarize both sets of labels
    print("[INFO] binarizing labels...")
    screensizeLB = LabelBinarizer()
    colorLB = LabelBinarizer()

    screenSizeLabels = screensizeLB.fit_transform(screen_size_label)
    colorLabels = colorLB.fit_transform(color_label)

    #train
    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    split = train_test_split(data, screenSizeLabels, colorLabels,
                             test_size=0.2, random_state=42)
    (trainX, testX, trainScreenSizeY, testScreenSizeY,
     trainColorY, testColorY) = split

    # data_dimension_checking(trainX, training_data=True, label=False)
    # data_dimension_checking(testX, training_data=False, label=False)
    # data_dimension_checking(trainScreenSizeY, training_data=True, label=True)
    # data_dimension_checking(testScreenSizeY, training_data=False, label=True)
    # data_dimension_checking(trainColorY, training_data=True, label=True)
    # data_dimension_checking(testColorY, training_data=False, label=True)

    #trainX = np.array(trainX)
    #testX = np.array(testX)
    #trainScreenSizeY = np.array(trainScreenSizeY)
    #testScreenSizeY = np.array(testScreenSizeY)
    #trainColorY = np.array(trainColorY)
    #testColorY = np.array(testColorY)

    class_for_color = 26
    class_for_screen_size = 6

    model= MobileNet.build(IMG_DIM,IMG_DIM,numCategories=len(screensizeLB.classes_),numColors=len(colorLB.classes_),finalAct="softmax")

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

    # plot the total loss, category loss, and color loss
    lossNames = ["loss", "screen_size_output_loss", "color_output_loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

    # loop over the loss names
    for (i, l) in enumerate(lossNames):
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