"""
@author - Sivaramakrishnan
"""
from data.load_data import data_loader
from models.basic_model import base_model
import tensorflow as tf
import numpy as np
TRAIN_FILE_NAME = "/Users/siva/Documents/falconai/training.txt"
BATCH_SIZE = 5

def train():
    train_data,train_labels,data_itr = data_loader(TRAIN_FILE_NAME).data_loader_main(BATCH_SIZE)
    base_model_config = {"epochs":10,\
                         "batch_size":10,\
                         "optimizer":tf.train.AdamOptimizer,\
                         "lr":0.001,\
                         "loss":tf.nn.softmax_cross_entropy_with_logits,\
                         }

    model = base_model(base_model_config,train_data,train_labels)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(data_itr.initializer)
        for _ in range(10):
            ACCURACY_LIST = []
            LOSS_LIST = []
            try:
                while True:
                    to_compute = [model.optimizer,model.calculated_acc,model.calculated_loss]
                    _,acc_,loss_ = sess.run(to_compute)
                    print(acc_,loss_)
                    ACCURACY_LIST.append(acc_)
                    LOSS_LIST.append(loss_)
            except:
                print("Completed One round of training")
                print(np.mean(ACCURACY_LIST))
                print(np.mean(LOSS_LIST))

if __name__ == "__main__":
    train()

