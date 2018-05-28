"""
@author - Sivaramakrishnan
"""
from data.load_data import data_loader
from models.basic_model import base_model
from models.regularized_model import regularized_model
import tensorflow as tf
import numpy as np
TRAIN_FILE_NAME = "/Users/siva/Documents/falconai/training.txt"

def train(base_model_config):
    with tf.device("/cpu:0"):
        data,labels,train_op,val_op = data_loader(TRAIN_FILE_NAME).data_loader_train(base_model_config["batch_size"])

    #--------------- Different models ----------------#
    #model_obj = base_model(base_model_config,data,labels)
    model_obj = regularized_model(base_model_config,data,labels)
    # ------------------------------------------------#
    model = model_obj.inference()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(base_model_config["epochs"]):
            print("*********** EPOCH %s ***********"%str(e))
            sess.run(train_op)
            model_obj.set_keep_probability(base_model_config["dropout"])
            model_obj.reset_train_bool(True)
            ACCURACY_LIST,LOSS_LIST,TEST_LOSS_LIST,TEST_ACCURACY_LIST = [],[],[],[]
            try:
                while True:
                    _,acc_,loss_ = sess.run([model["optimizer"],model["acc"],model["loss"]])
                    ACCURACY_LIST.append(acc_)
                    LOSS_LIST.append(loss_)
            except:
                print("Train Accuracy:",np.mean(ACCURACY_LIST),"|","Train Loss:",np.mean(LOSS_LIST))
            sess.run(val_op)
            model_obj.set_keep_probability(1.0)
            model_obj.reset_train_bool(False)
            try:
                while(True):
                    test_acc,test_loss = sess.run([model["acc"],model["loss"]])
                    TEST_ACCURACY_LIST.append(test_acc)
                    TEST_LOSS_LIST.append(test_loss)
            except:
                print("Val Accuracy:", np.mean(TEST_ACCURACY_LIST), "|", "Val Loss:", np.mean(TEST_LOSS_LIST))

if __name__ == "__main__":
    base_model_config = {"epochs": 30, \
                         "batch_size": 52, \
                         "optimizer": tf.train.AdamOptimizer, \
                         "lr": 0.0001, \
                         "lambda_":0.001,\
                         "loss": tf.nn.softmax_cross_entropy_with_logits, \
                         "dropout": 0.8
                         }

    train(base_model_config)

