"""
@author - Sivaramakrishnan
"""

import tensorflow as tf
import numpy as np
from data.load_data import data_loader

#ROOT_PATH = "/Users/siva/Documents/falconai/"
ROOT_PATH = "/home/scelesticsiva/Documents/falconai/"
TRAIN_FILE_NAME = ROOT_PATH+"training.txt"
VGG_WEIGHTS_FILE = ROOT_PATH+"classification_of_fashion_category/pre_trained/vgg16.npy"

names,data,vgg_features,labels,train_op,val_op = data_loader(TRAIN_FILE_NAME).data_loader_train(50,["/cpu:0","/gpu:0"],VGG_WEIGHTS_FILE,True)

init = tf.global_variables_initializer()

numpy_features = []
numpy_labels = []
numpy_names = []
with tf.Session() as sess:
    sess.run(init)
    sess.run(train_op)
    try:
        while True:
            features_,labels_,names_ = sess.run([vgg_features,labels,names])
            numpy_features += features_.tolist()
            numpy_labels += np.argmax(labels_,axis = 1).tolist()
            numpy_names += names_.tolist()
    except:
        print("Finished extracting train features")

    sess.run(val_op)
    try:
        while True:
            features_,labels_,names_ = sess.run([vgg_features,labels,names])
            numpy_features += features_.tolist()
            numpy_labels += np.argmax(labels_,axis = 1).tolist()
            numpy_names += names_.tolist()
    except:
        print("Finished extract validation features")
    final_dict_ = {"features":np.array(numpy_features),"labels":np.array(numpy_labels),\
                   "file_names":np.array(numpy_names)}
    np.save("vgg_features",final_dict_)

