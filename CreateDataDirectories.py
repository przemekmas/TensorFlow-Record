import glob
import shutil
import os
import csv
from PIL import Image
 
src_dir = "/home/linux/Desktop/CharacterTFRecord/CreateCorrespondingFolders/data/{0}.Bmp"
dst_dir = "/home/linux/Desktop/CharacterTFRecord/CreateCorrespondingFolders/{0}"
labels_dir = "/home/linux/Desktop/CharacterTFRecord/CreateCorrespondingFolders/labels.csv"
directories = ""

with open(labels_dir) as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		if not os.path.exists("/home/linux/Desktop/CharacterTFRecord/CreateCorrespondingFolders/datafolders/{0}".format(row['Class'])):
    			os.makedirs("/home/linux/Desktop/CharacterTFRecord/CreateCorrespondingFolders/datafolders/{0}".format(row['Class']))
			directories += format(row['Class']) + "\n"
		
		img = Image.open("/home/linux/Desktop/CharacterTFRecord/CreateCorrespondingFolders/data/{0}.Bmp".format(row['ID']))				
		img.save("/home/linux/Desktop/CharacterTFRecord/CreateCorrespondingFolders/datafolders/{0}/{1}".format(row['Class'], row['ID']),'png')

#shutil.copy(src_dir.format(row['ID']), "/home/linux/Desktop/CharacterTFRecord/CreateCorrespondingFolders/datafolders/{0}/{1}.Bmp".format(row['Class'], row['ID']))

print(directories)
print('Created folders and files corresponding to the folder names')
