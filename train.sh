#!/bin/bash
#
# Author - Sivaramakrishnan
#
#The main script to test the pre trained model
VGG_PRETRAINED_WEIGHTS="https://www.dropbox.com/s/o0t8wuqbu64eazh/vgg16.npy?dl=1"
VGG_FILE="vgg16.npy"

echo "Downloading VGG16 weights file"
curl -L -o vgg16.npy $VGG_PRETRAINED_WEIGHTS

echo "Running the training script"
#tail +3 "$1" > "$1" #to remove the first two rows in the text file which contains number of images and the string "image category label"

if [ "$2" == 'use_gpu' ] ; then
    python3 train.py --train_labels "$1" --run_in_gpu True --vgg_pretrained_weights $VGG_FILE
else
    python3 train.py --train_labels "$1" --vgg_pretrained_weights $VGG_FILE
fi


