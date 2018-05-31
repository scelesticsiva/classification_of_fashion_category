"""
@author - Sivaramakrishnan
"""
import tensorflow as tf
import numpy as np
from data.load_data import data_loader
from models.model_with_vgg_features import vgg_features_model
from utils import evaluation_metrics
import argparse

def test(model_config,TEST_FILE_NAME,VGG_WEIGHTS_FILE):
    _,data,vgg_features,labels,test_op = data_loader(TEST_FILE_NAME,True).data_loader_test(model_config["batch_size"],\
                                                                                      model_config["devices"],\
                                                                                      VGG_WEIGHTS_FILE,\
                                                                                      model_config["use_vgg_features"])
    model_obj = vgg_features_model(model_config,data,vgg_features,labels)
    model = model_obj.inference()

    restorer = tf.train.Saver()

    with tf.Session() as sess:
        restorer.restore(sess,model_config["model_dir"]+"/checkpoint.ckpt")
        sess.run(test_op)
        ACCURACY_LIST,LOSS_LIST,MLABELS,MPREDICTIONS = [],[],[],[]
        try:
            while True:
                feed_dict_ = {model_obj.train_bool:0,model_obj.keep_probability:1.0}
                acc_,loss_,m_labels,m_predictions = sess.run([model["acc"],model["loss"],model["labels"],\
                                                              model["predictions"]],feed_dict=feed_dict_)
                ACCURACY_LIST.append(acc_)
                LOSS_LIST.append(loss_)
                MLABELS += m_labels.tolist()
                MPREDICTIONS += m_predictions.tolist()
        except:
            print("Done testing","\n")
            eval = evaluation_metrics.metrics(MLABELS,MPREDICTIONS)
            print("********** Testing Results ***********")
            print("Accuracy:",np.mean(ACCURACY_LIST),"|","Loss:",np.mean(LOSS_LIST))
            print("Precision:",eval.precision)
            print("Recall:",eval.recall)
            print("F1 score:",eval.f1score)
            print("Confusion Matrix:","\n")
            print(eval.confusion_mat,"\n")
            print(eval.correct_predictions)
            print("**************************************")



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--test_labels",type = str,\
                                 help="Path to the test labels file")
    argument_parser.add_argument("--run_in_gpu",type = bool,default=False,\
                                 help="If you installed tensorflow GPU you can turn this to True")
    argument_parser.add_argument("--vgg_pretrained_weights",type = str,\
                                 help = "Path to VGG weights file")
    args = argument_parser.parse_args()
    if args.run_in_gpu:
        devices = ["/cpu:0","/gpu:0"]
    else:
        devices = ["/cpu:0","/cpu:0"]

    model_config = {"batch_size": 26, \
                     "optimizer": tf.train.AdamOptimizer, \
                     "lr": 0.0001, \
                     "lambda_":0.001,\
                     "loss": tf.nn.softmax_cross_entropy_with_logits, \
                     "use_vgg_features":True,\
                     "model_dir":"checkpoints_vgg_features",\
                     "devices":devices,\
                     }
    test(model_config,args.test_labels,args.vgg_pretrained_weights)
