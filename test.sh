#!/bin/bash

#The main script to test the pre trained model
TRAINED_MODEL_URL="https://www.dropbox.com/s/dzcymeotbkb6i5g/trained_model.tar.gz?dl=1"
VGG_PRETRAINED_WEIGHTS="https://www.dropbox.com/s/o0t8wuqbu64eazh/vgg16.npy?dl=1"
VGG_FILE="vgg16.npy"

echo "Downloading the model"
curl -L -o trained_model.tar.gz $TRAINED_MODEL_URL

echo "Untarring the model"
tar -xzvf trained_model.tar.gz

echo "Downloading VGG16 weights file"
curl -L -o vgg16.npy $VGG_PRETRAINED_WEIGHTS

echo "Running the testing script"
#tail +3 "$1" > "$1" #to remove the first two rows in the text file which contains number of images and the string "image category label"

if [ "$2" == 'use_gpu' ] ; then
    python3 test.py --test_labels "$1" --run_in_gpu True --vgg_pretrained_weights $VGG_FILE
else
    python3 test.py --test_labels "$1" --vgg_pretrained_weights $VGG_FILE
fi

echo $GPU



