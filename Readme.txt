-Additional Information

The create corresponding folder contains a sub folder with images and a labels.csv file.
The labels.csv file contains the file name and its corresponding label.
The "CreateDataDirectories.py" file has been created to seperate the images into subfolders with corresponding names.
The "CreateDataDirectories.py" file does not need to be if you have data/images setup in your subfolders.
The data can also be created separately and shoved into subdirectories.
The directory already contains files/data which has been setup for characters and numbers.
The current project already contains ready data.

-Create TFRecord Steps

1. Make sure that your image directory has subfolders which have a name that match your images. For example the "ImageData" folder contains sub folders where the images are stored. The sub folder "0" contains all the images with a number 0. The sub folder "1" contains all the images of number 1 and so on.

2. Once all of your images have been placed in sub directories, you will need to create a "folderlabel.txt" file, which should specify all your label names such as "A" or "B". Use the current "folderlabel.txt" file as an example.

3. Afterwards, you can finally run the "CreateTFRecord.py" file. This will create all of your binary TFRecord files in your current directory which will store your images in binary format.

4. You have now successfully created your TFRecords. You can now read the data/images from the TFRecords by running the "ReadDataFromTFRecord.py" file, which will read the images from you TFRecord and save them in a folder "resized_image" in your current directory.


