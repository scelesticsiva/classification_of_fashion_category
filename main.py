"""
@author - Sivaramakrishnan
"""
from data.load_data import data_loader
from models.basic_model import base_model
import tensorflow as tf
import numpy as np
TRAIN_FILE_NAME = "/Users/siva/Documents/falconai/training.txt"
BATCH_SIZE = 128

def train():
    train_data,train_labels,train_op,val_op = data_loader(TRAIN_FILE_NAME).data_loader_train(BATCH_SIZE)
    base_model_config = {"epochs":10,\
                         "batch_size":10,\
                         "optimizer":tf.train.AdamOptimizer,\
                         "lr":0.001,\
                         "loss":tf.nn.softmax_cross_entropy_with_logits,\
                         }

    model_obj = base_model(base_model_config,train_data,train_labels)
    model = model_obj.inference()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(20):
            sess.run(train_op)
            ACCURACY_LIST = []
            LOSS_LIST = []
            TEST_ACCURACY_LIST = []
            TEST_LOSS_LIST = []
            try:
                count = 0
                while True:
                    count += 1
                    to_compute = [model["optimizer"],model["acc"],model["loss"]]
                    _,acc_,loss_ = sess.run(to_compute)
                    if count % 50 == 0:
                        print(acc_,loss_)
                    ACCURACY_LIST.append(acc_)
                    LOSS_LIST.append(loss_)
            except:
                print("*********** Training ************")
                print(np.mean(ACCURACY_LIST))
                print(np.mean(LOSS_LIST))
            sess.run(val_op)
            try:
                count = 0
                while(True):
                    count += 1
                    to_compute = [model["acc"],model["loss"]]
                    test_acc,test_loss = sess.run(to_compute)
                    if count % 50 == 0:
                        print(test_acc,test_loss)
                    TEST_ACCURACY_LIST.append(test_acc)
                    TEST_LOSS_LIST.append(test_loss)
            except:
                print("*********** Validation ************")
                print(np.mean(TEST_ACCURACY_LIST))
                print(np.mean(TEST_LOSS_LIST))

if __name__ == "__main__":
    train()

