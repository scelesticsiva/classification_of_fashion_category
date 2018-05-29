"""
@author - Sivaramakrishnan
"""
from data.load_data import data_loader
from models.basic_model import base_model
from models.regularized_model import regularized_model
from models.model_with_vgg_features import vgg_features_model
from models.simple_vgg_features_model import simple_vgg_features_model
import tensorflow as tf
import numpy as np

ROOT_PATH = "/Users/siva/Documents/falconai/"
TRAIN_FILE_NAME = ROOT_PATH+"training.txt"
VGG_WEIGHTS_FILE = ROOT_PATH+"classification_of_fashion_category/pre_trained/vgg16.npy"

def train(base_model_config):
    data,vgg_features,labels,train_op,val_op = data_loader(TRAIN_FILE_NAME).data_loader_train(base_model_config["batch_size"],\
                                                                                              base_model_config["devices"], \
                                                                                              VGG_WEIGHTS_FILE, \
                                                                                              base_model_config["use_vgg_features"])

    #--------------- Different models ----------------#
    #model_obj = base_model(base_model_config,data,labels)
    model_obj = regularized_model(base_model_config,data,labels)
    #model_obj = vgg_features_model(base_model_config,data,vgg_features,labels)
    #model_obj = simple_vgg_features_model(base_model_config,data,vgg_features,labels)
    # ------------------------------------------------#
    model = model_obj.inference()
    init = tf.global_variables_initializer()

    if base_model_config["checkpoint"]:
        saver = tf.train.Saver()


    with tf.Session() as sess:
        train_saver = tf.summary.FileWriter(base_model_config["summary_dir"] + "/train",sess.graph)
        test_saver = tf.summary.FileWriter(base_model_config["summary_dir"] + "/test",sess.graph)

        sess.run(init)
        global_step_train,global_step_test = 0,0
        for e in range(base_model_config["epochs"]):
            print("*********** EPOCH %s ***********"%str(e))
            sess.run(train_op)
            ACCURACY_LIST,LOSS_LIST,TEST_LOSS_LIST,TEST_ACCURACY_LIST = [],[],[],[]
            try:
                while True:
                    global_step_train += 1
                    feed_dict_ = {model_obj.keep_probability:base_model_config["dropout"],model_obj.train_bool:1}
                    _,acc_,loss_,summ_ = sess.run([model["optimizer"],model["acc"],model["loss"],model["summary"]],feed_dict=feed_dict_)
                    ACCURACY_LIST.append(acc_)
                    LOSS_LIST.append(loss_)
                    train_saver.add_summary(summ_,global_step=global_step_train)
            except:
                print("Train Accuracy:",np.mean(ACCURACY_LIST),"|","Train Loss:",np.mean(LOSS_LIST))
            sess.run(val_op)
            try:
                while(True):
                    global_step_test += 1
                    feed_dict_ = {model_obj.keep_probability:1.0, model_obj.train_bool: 0}
                    test_acc,test_loss,test_summ_ = sess.run([model["acc"],model["loss"],model["summary"]],feed_dict=feed_dict_)
                    TEST_ACCURACY_LIST.append(test_acc)
                    TEST_LOSS_LIST.append(test_loss)
                    test_saver.add_summary(test_summ_,global_step=global_step_test)
            except:
                print("Val Accuracy:", np.mean(TEST_ACCURACY_LIST), "|", "Val Loss:", np.mean(TEST_LOSS_LIST))

            if base_model_config["checkpoint"]:
                print("Saving model...")
                saver.save(sess,base_model_config["model_dir"]+"/checkpoint.ckpt")

if __name__ == "__main__":
    base_model_config = {"epochs": 30, \
                         "batch_size": 52, \
                         "optimizer": tf.train.AdamOptimizer, \
                         "lr": 0.0001, \
                         "lambda_":0.001,\
                         "loss": tf.nn.softmax_cross_entropy_with_logits, \
                         "dropout": 0.6,\
                         "use_vgg_features":True,\
                         "checkpoint":True,\
                         "model_dir":ROOT_PATH+"checkpoints_try",\
                         "devices":["/cpu:0","/cpu:0"],\
                         "summary_dir":ROOT_PATH+"tensorboard_try"
                         }

    train(base_model_config)

