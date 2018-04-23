import glob
import shutil
import os
import csv
from PIL import Image

labels_dir = "/home/linux/Desktop/CharacterTFRecord/labels.csv"

def CreateTrainData():
	with open(labels_dir) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if not os.path.exists("/home/linux/Desktop/CharacterTFRecord/train/{0}".format(row['Class'])):
					os.makedirs("/home/linux/Desktop/CharacterTFRecord/train/{0}".format(row['Class']))
					#directories += format(row['Class']) + "\n"

			img = Image.open("/home/linux/Desktop/CharacterTFRecord/data/{0}.Bmp".format(row['ID']))				
			img.save("/home/linux/Desktop/CharacterTFRecord/train/{0}/{1}".format(row['Class'], row['ID']),'png')
			
def CreateTestData():
	with open(labels_dir) as csvfile:
		reader = csv.DictReader(csvfile)
		count = 0
		for row in reader:
			count = count + 1
			if (count == 15):
				count = 0
				if not os.path.exists("/home/linux/Desktop/CharacterTFRecord/test/{0}".format(row['Class'])):
						os.makedirs("/home/linux/Desktop/CharacterTFRecord/test/{0}".format(row['Class']))
						#directories += format(row['Class']) + "\n"

				img = Image.open("/home/linux/Desktop/CharacterTFRecord/data/{0}.Bmp".format(row['ID']))				
				img.save("/home/linux/Desktop/CharacterTFRecord/test/{0}/{1}".format(row['Class'], row['ID']),'png')

CreateTrainData()
CreateTestData()
print('Created folders and files corresponding to the folder names')
