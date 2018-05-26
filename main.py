"""
@author - Sivaramakrishnan
"""
from data.load_data import data_loader
from models.basic_model import base_model
import tensorflow as tf
TRAIN_FILE_NAME = "/Users/siva/Documents/falconai/training.txt"

def train():
    train_data_loader = data_loader(TRAIN_FILE_NAME).data_loader_main(5)

    base_model_config = {"epochs":10,\
                         "batch_size":10,\
                         "optimizer":tf.train.AdamOptimizer,\
                         "lr":0.0001,\
                         "loss":tf.nn.softmax_cross_entropy_with_logits,\
                         }

    model = base_model(base_model_config).inference()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        img,label = sess.run(train_data_loader)
        print(img.shape)
        print(label)

if __name__ == "__main__":
    train()

