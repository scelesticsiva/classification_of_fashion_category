"""
@author - Sivaramakrishnan
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from pre_trained import pre_trained_vgg

class data_loader(object):
    def __init__(self,filename,test = False):
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
        if not test:
            self.train_data,self.val_data,self.train_labels,self.val_labels = train_test_split(self.list_of_images,self.list_of_labels)
            train_args = np.argmax(self.train_labels,axis = 1)
            val_args = np.argmax(self.val_labels,axis = 1)
            counter_train = Counter(train_args)
            counter_val = Counter(val_args)
            print("************ Data Statistics *************")
            print("Training:",sorted([(label_dict_rev[each],counter_train[each]) for each in counter_train]))
            print("Validation:",sorted([(label_dict_rev[each], counter_val[each]) for each in counter_val]))
            print("******************************************")
        if test:
            test_args = np.argmax(self.list_of_labels,axis = 1)
            counter_test = Counter(test_args)
            print("*********** Test Data Statistics ***********")
            print("Test Data:",sorted([(label_dict_rev[each],counter_test[each]) for each in counter_test]))
            print("********************************************")

    def global_contrast_normalization(self,image):
        mean = tf.reduce_mean(image)
        image_mean_centered = tf.subtract(image, mean)
        var = tf.maximum(tf.constant([1e-7]),
                         tf.sqrt(tf.reduce_mean(tf.square(image_mean_centered))))
        return tf.divide(image_mean_centered, var)

    def _parse_function_val(self,img_name,label):
        """
        Reading the img from the file name
        :param img_name: [tensor] containing all the images file names
        :param label: [tensor] containing all the labels
        :return: [tuple] images preprocessed and labels
        """
        image_string = tf.read_file(img_name)
        image_decoded = tf.image.decode_image(image_string)
        image_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        #image_normalized = self.global_contrast_normalization(image_float)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_float, 224, 224)
        return img_name,image_resized, label

    def _parse_function_train(self,img_name,label):
        """
        Reading the img from the file name
        :param img_name: [tensor] containing all the images file names
        :param label: [tensor] containing all the labels
        :return: [tuple] images preprocessed and labels
        """
        image_string = tf.read_file(img_name)
        image_decoded = tf.image.decode_image(image_string)
        image_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        #image_normalized = self.global_contrast_normalization(image_float)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_float, 224, 224)
        with tf.name_scope("image_preprocessing_train"):
            image = tf.image.random_brightness(image_resized, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        return img_name,image,label

    def data_loader_train(self,BATCH_SIZE,devices,VGG_WEIGHTS_FILE,use_pretained_vgg = False,buffer_size = 104):
        """
        Main function that created train data iterator
        :param BATCH_SIZE: [int] Batch size required
        :return: [tensor] next batch of data
        """
        self.filenames = tf.constant(self.train_data)
        self.categories = tf.constant(self.train_labels)

        self.val_filenames = tf.constant(self.val_data)
        self.val_categories = tf.constant(self.val_labels)

        with tf.device(devices[0]):
            train_dataset = tf.data.Dataset.from_tensor_slices((self.filenames,self.categories))
            train_dataset = train_dataset.map(self._parse_function_train)
            train_dataset = train_dataset.batch(BATCH_SIZE)
            train_dataset = train_dataset.prefetch(buffer_size=buffer_size)

            val_dataset = tf.data.Dataset.from_tensor_slices((self.val_data,self.val_categories))
            val_dataset = val_dataset.map(self._parse_function_val)
            val_dataset = val_dataset.batch(BATCH_SIZE)
            val_dataset = val_dataset.prefetch(buffer_size=buffer_size)

            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

            next_names,next_data,next_labels = iterator.get_next()

        if use_pretained_vgg:
            #with tf.device(devices[1]):
            vgg_features = pre_trained_vgg.Vgg16(devices,VGG_WEIGHTS_FILE).build(next_data)
        else:
            vgg_features = None

        train_op = iterator.make_initializer(train_dataset)
        val_op = iterator.make_initializer(val_dataset)

        return [next_names,next_data,vgg_features,next_labels,train_op,val_op]

    def data_loader_test(self,BATCH_SIZE,devices,VGG_WEIGHTS_FILE,use_pretrained_vgg = False,buffer_size = 104):
        """
        Test data loader
        :param BATCH_SIZE: [int] Batch size to be used for fetching test data
        :param devices: [list] list of devices that are available
        :param VGG_WEIGHTS_FILE: [str] path to the vgg net weights file
        :param use_pretrained_vgg: [bool] whether to use vgg net
        :param buffer_size: [int] load data onto buffer to utilize GPU to the max
        :return:
        """
        self.test_file_names = tf.constant(self.list_of_images)
        self.test_categories = tf.constant(self.list_of_labels)

        with tf.device(devices[0]):
            test_dataset = tf.data.Dataset.from_tensor_slices((self.test_file_names,self.test_categories))
            test_dataset = test_dataset.map(self._parse_function_val)
            test_dataset = test_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.prefetch(buffer_size=buffer_size)

            test_iterator = tf.data.Iterator.from_structure(test_dataset.output_types,test_dataset.output_shapes)

            test_next_names,test_next_data,test_next_labels = test_iterator.get_next()

        if use_pretrained_vgg:
            #with tf.device(devices[1]):
            test_vgg_features = pre_trained_vgg.Vgg16(devices,VGG_WEIGHTS_FILE).build(test_next_data)
        else:
            test_vgg_features = None

        test_op = test_iterator.make_initializer(test_dataset)

        return [test_next_names,test_next_data,test_vgg_features,test_next_labels,test_op]