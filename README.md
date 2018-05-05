# TensorFlow Training Custom Dataset

This GitHub repository aims to help with feeding a custom dataset through a convolutional neural network by the use of TensorFlow. The repository includes all the necessary steps for feeding a custom dataset through a convolutional neural network. The repository already includes a custom dataset as an example of how the data should be setup. Additionally, the steps for feeding a custom dataset will include creating a TFRecord which contains the image data and the corresponding labels. The TFRecord files are in binary format and will be used to feed batches of data into the convolutional neural network.

## Main Steps

### Beginner Tips

* If you are creating the TFRecords and custom data by using the current scripts. Please make sure that the folder structure stays the same.
* If you want to use your own data to create the TFRecords you will need to follow the steps provided.
* The training process is only setup for images with a width and height of 28 by 28 pixels.
* The current repository will allow you to re-run the scripts with the existing example images.

### Setting up the data

* To get started you will need to setup your custom dataset and place all of your images within a data folder and number them starting from 1. (The current data folder demonstrates how your data should be labelled)
* After all your images have been placed and numbered within the datafolder. You will need to create a CSV file which should include two columns ID and Class. The id represents the image name and the class is the corresponding label for that image. (The exsisting labels.csv file can be used as an example)
* Furthermore, an additional text file is needed that specifies all existing labels. (The existing imagelabels.txt file can be used as an example)
* The next step is to run the CreateDataDirectories.py script which will create a train and test folders. The train and test folders will contain subfolders that are labelled and contain your corresponding images.
* You have now successfully setup your data! 

### Creating a TFRecord

* To create your TFRecord you will need to run the CreateTFRecord.py script.
* After the python script is successfully executed it will create a train and validation binary files (These are your tfrecord files which will be created in the current directory).
* One of the TFRecord files will contain all of your training data and the other will contain validation data.
* You have now created your TFRecords successfully.

### Training

* In order to train your data which is now stored in your TFRecords. You will need to run the TrainCustomDataset.py script.
* The training process will now begin and you will be able to see the training accuracy.
* Also, the graph will be saved in ckpt format.
* You have successfully trained a custom dataset by the use of TensorFlow.

### Additional Information

* You can check the images which are stored in your TFRecord by running the ReadDataFromTFRecord.py script. However, you will need to modify the script to read your chosen TFRecord file by providing/changing the TFRecord filename that the script is trying to read from.

## References

The links below have been used to get some of the code for this repository

* https://github.com/yeephycho/tensorflow_input_image_by_tfrecord
* https://github.com/sujayVittal/Machine-Learning-with-TensorFlow-Study-Jam-2017/blob/master/mnist-advanced.py
