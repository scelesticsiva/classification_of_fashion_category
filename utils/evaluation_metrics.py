"""
@author - Sivaramakrishnan
"""
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix

class metrics(object):
    """
    Class which calculates different metrics which can be extended to include more metrics
    """
    def __init__(self,labels,predictions):
        self.labels = np.argmax(labels,axis = 1)
        self.predictions = np.argmax(predictions,axis = 1)
        self.label_dict_rev = {0: "Jeans", 1: "Sweatpants", 2: "Blazer"}
        self.calculate_accuracy()
        self.calculate_other_metrics()
        self.calculate_confusion_mat()
        self.calculate_correct_predictions()

    def calculate_accuracy(self):
        self.acc = np.mean(np.equal(self.labels,self.predictions))

    def calculate_other_metrics(self):
        self.f1score = f1_score(self.labels,self.predictions,average="macro")
        self.recall = recall_score(self.labels,self.predictions,average="macro")
        self.precision = precision_score(self.labels,self.predictions,average="macro")

    def calculate_confusion_mat(self):
        self.confusion_mat = confusion_matrix(self.labels,self.predictions)

    def calculate_correct_predictions(self):
        """
        Calculating how many images got correctly classified in each class
        :return: None
        """
        correct_indices = np.equal(self.labels,self.predictions)
        correct_predictions = self.labels[correct_indices]
        count_correct_predictions = Counter(correct_predictions)
        count_labels = Counter(self.labels)
        self.correct_predictions = ""
        for each in count_labels:
            if each in count_correct_predictions:
                self.correct_predictions += "( {0} out of {1} {2})".format(str(count_correct_predictions[each]),\
                                                         str(count_labels[each]),self.label_dict_rev[each])
