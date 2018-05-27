"""
@author - Sivaramakrishnan
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

class data_loader(object):
    def __init__(self,filename):
        """
        :param filename: [str] which contains all the file paths to images
        """
        self.list_of_images, self.list_of_labels = [], []
        label_dict = {"Jeans": 0, "Sweatpants": 1, "Blazer": 2}
        label_dict_rev = {0:"Jeans",1:"Sweatpants", 2:"Blazer"}
        number_of_labels = len(label_dict)
        with open(filename) as file:
            for each in file:
                file_name = each.split(" ")[0]
                self.list_of_images.append(file_name)
                self.list_of_labels.append([np.eye(number_of_labels)[label_dict[i]] for i in label_dict if i in file_name][0])
        self.list_of_images = np.array(self.list_of_images)
        self.list_of_labels = np.array(self.list_of_labels)
        self.train_data,self.val_data,self.train_labels,self.val_labels = train_test_split(self.list_of_images,self.list_of_labels)
        train_args = np.argmax(self.train_labels,axis = 1)
        val_args = np.argmax(self.val_labels,axis = 1)
        counter_train = Counter(train_args)
        counter_val = Counter(val_args)
        print("************ Data Statistics *************")
        print("Training:",sorted([(label_dict_rev[each],counter_train[each]) for each in counter_train]))
        print("Validation:",sorted([(label_dict_rev[each], counter_val[each]) for each in counter_val]))
        print("******************************************")


    def _parse_function(self,img_name,label):
        """
        Reading the img from the file name
        :param img_name: [tensor] containing all the images file names
        :param label: [tensor] containing all the labels
        :return: [tuple] images preprocessed and labels
        """
        image_string = tf.read_file(img_name)
        image_decoded = tf.image.decode_image(image_string)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 224, 224)
        image_resized = tf.image.convert_image_dtype(image_resized,dtype = tf.float32)
        return image_resized, label

    def data_loader_train(self,BATCH_SIZE):
        """
        Main function that created train data iterator
        :param BATCH_SIZE: [int] Batch size required
        :return: [tensor] next batch of data
        """
        self.filenames = tf.constant(self.train_data)
        self.categories = tf.constant(self.train_labels)

        self.val_filenames = tf.constant(self.val_data)
        self.val_categories = tf.constant(self.val_labels)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.filenames,self.categories))
        train_dataset = train_dataset.map(self._parse_function)
        train_dataset = train_dataset.batch(BATCH_SIZE)

        val_dataset = tf.data.Dataset.from_tensor_slices((self.val_data,self.val_categories))
        val_dataset = val_dataset.map(self._parse_function)
        val_dataset = val_dataset.batch(BATCH_SIZE)


        #data_iterator = dataset.make_initializable_iterator()
        #next_data,next_labels = data_iterator.get_next()

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

        next_data,next_labels = iterator.get_next()

        train_op = iterator.make_initializer(train_dataset)
        val_op = iterator.make_initializer(val_dataset)

        return [next_data,next_labels,train_op,val_op]