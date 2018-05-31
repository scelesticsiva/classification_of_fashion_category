# classification_of_fashion_category
This is an attempt to classify images based on the fashion items present
in an image. I used a combination of pre-trained VGG network and CNN to acheive 
a validation accuracy of 88%.

To install all the dependencies run the following command,
```bash
pip3 install -r requirements.txt
```

If you want to run the model in GPU, you need tensorflow for GPU installed.
To install tensorflow GPU version follow the instructions [here](https://www.tensorflow.org/install/install_linux)

## Using pre - trained model
---

To use my pre-trained model, you need to have a folder of images and a text file containing
path to each of those images with the label in their names,

Labels file can look something like this(you do not need to worry about the category labels,because I am taking the labels from 
the file name - in this example the label of the image is "Jeans"),
```
89222
image_name  category_label
img/Mineral_Wash_Skinny_Jeans/img_00000001.jpg                         26
```
To use a pretained model, run the following code with the test labels file as command
line argument (make sure that the path in the labels files either contains the full path to the images
or relative path to this directory),
```bash
./test.sh <path to test labels file>
``` 
if you have tensorflow GPU installed, you can run the following script,
```bash
./test.sh <path to test labels file> use_gpu
```



