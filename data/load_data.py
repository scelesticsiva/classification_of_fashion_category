"""
@author - Sivaramakrishnan
"""

import tensorflow as tf
import numpy as np

class data_loader(object):
    def __init__(self,filename):
        """
        :param filename: [str] which contains all the file paths to images
        """
        self.list_of_images, self.list_of_labels = [], []
        label_dict = {"Jeans": 0, "Sweatpants": 1, "Blazer": 2}
        number_of_labels = len(label_dict)
        with open(filename) as file:
            for each in file:
                file_name = each.split(" ")[0]
                self.list_of_images.append(file_name)
                self.list_of_labels.append([np.eye(number_of_labels)[label_dict[i]] for i in label_dict if i in file_name][0])
        self.list_of_images = np.array(self.list_of_images)
        self.list_of_labels = np.array(self.list_of_labels)

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

    def data_loader_main(self,BATCH_SIZE):
        """
        Main function that created data iterator
        :param BATCH_SIZE: [int] Batch size required
        :return: [tensor] next batch of data
        """
        self.filenames = tf.constant(self.list_of_images)
        self.labels = tf.constant(self.list_of_labels)

        dataset = tf.data.Dataset.from_tensor_slices((self.filenames,self.labels))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.batch(BATCH_SIZE)

        #data_iterator = dataset.make_one_shot_iterator()
        data_iterator = dataset.make_initializable_iterator()
        next_data,next_labels = data_iterator.get_next()

        return [next_data,next_labels,data_iterator]