"""
@author - Sivaramakrishnan
"""
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix

class metrics(object):
    def __init__(self,labels,predictions):
        self.labels = np.argmax(labels,axis = 1)
        self.predictions = np.argmax(predictions,axis = 1)

    def calculate_accuracy(self):
        self.acc = np.reduce_mean(np.equal(self.labels,self.predictions))

    def calculate_other_metrics(self):
        self.f1score = f1_score(self.labels,self.predictions)
        self.recall = recall_score(self.labels,self.predictions)
        self.precision = precision_score(self.labels,self.predictions)

    def calculate_confusion_mat(self):
        self.confusion_mat = confusion_matrix(self.labels,self.predictions)