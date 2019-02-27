# USAGE
# python train.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.minivggnet import MiniVGGNet
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils import paths
import argparse
import random
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

IMAGE_DIMS = (64, 64, 3)

def csv_image_generator(inputPath, bs, mlb, mode="train", aug=None):
	# open the CSV file for reading
	f = open(inputPath, "r")
	IMAGE_DIMS = (64, 64, 3)
	# loop indefinitely	
	while True:
		# initialize our batches of images and labels
		images = []
		labels = []

		# keep looping until we reach our batch size
		while len(images) < bs:
			# attempt to read the next line of the CSV file
			line = f.readline()

			# check to see if the line is empty, indicating we have
			# reached the end of the file
			if line == "":
				# reset the file pointer to the beginning of the file
				# and re-read the line
				f.seek(0)
				line = f.readline()

				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
				if mode == "eval":
					break

			# extract the label and construct the image
			line = line.strip().split(",")
			label = line[1]
			label = [line[0],line[1]]
			#print(label)
			#image = cv2.imread(line[2])
			#image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
			image = np.array([int(x) for x in line[2:]], dtype="uint8")
			image = image.reshape((64, 64, 3))
			image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
			#image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
			image = img_to_array(image)
			# update our corresponding batches lists
			images.append(image)
			labels.append(label)

		# scale the raw pixel intensities to the range [0, 1]
		images = np.array(images, dtype="float") / 255.0
		# one-hot encode the labels		
		labels = np.array(labels)
		labels = mlb.transform(labels)
		# loop over each of the possible class labels and show them
		#for (i, label) in enumerate(mlb.classes_):
		#	print("{}. {}".format(i + 1, label))

		#print(labels)
		#labels = lb.transform(labels)
		# loop over each of the possible class labels and show them
		#for (i, label) in enumerate(mlb.classes_):
		#	print("{}. {}".format(i + 1, label))

		# if the data augmentation object is not None, apply it
		if aug is not None:
			(images, labels) = next(aug.flow(images,
				labels, batch_size=bs))

		# yield the batch to the calling function
		yield (images, labels)

# initialize the paths to our training and testing CSV files
TRAIN_CSV = "flowers17_training.csv"
TEST_CSV = "flowers17_testing.csv"

# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 20
EPOCHS = 20
BS = 32
INIT_LR = 1e-3
# initialize the total number of training and testing image
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

# open the training CSV file, then initialize the unique set of class
# labels in the dataset along with the testing labels
f = open(TRAIN_CSV, "r")
#labels = set()
labels = []
testLabels = []

# loop over all rows of the CSV file
for line in f:
	# extract the class label, update the labels list, and increment
	# the total number of training images
	label = line.strip().split(",")[0]
	label1 = line.strip().split(",")[1]
	label = [label,label1]
	labels.append(label)
	#labels.add(label)
	#labels.add(label1)
	NUM_TRAIN_IMAGES += 1

# close the training CSV file and open the testing CSV file
f.close()
f = open(TEST_CSV, "r")

# loop over the lines in the testing file
for line in f:
	# extract the class label, update the test labels list, and
	# increment the total number of testing images
	label = line.strip().split(",")[0]
	label1 = line.strip().split(",")[1]
	label = [label,label1]
	testLabels.append(label)
	NUM_TEST_IMAGES += 1

# close the testing CSV file
f.close()

# create the label binarizer for one-hot encoding labels, then encode
# the testing labels
#labels = np.array(labels)
#testLabels = np.array(testLabels)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
testLabels = mlb.fit_transform(testLabels)

#mlb = MultiLabelBinarizer()
# one-hot encode the labels
#mlb.fit(list(labels))
#testLabels = mlb.fit_transform(testLabels)

#lb = LabelBinarizer()
#lb.fit(list(labels))
#testLabels = lb.transform(testLabels)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# initialize both the training and testing image generators
trainGen = csv_image_generator(TRAIN_CSV, BS, mlb,
	mode="train", aug=aug)
testGen = csv_image_generator(TEST_CSV, BS, mlb,
	mode="train", aug=None)

# initialize our Keras model and compile it
#model = MiniVGGNet.build(64, 64, 3, len(lb.classes_))
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], 
	height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2],
	classes=len(mlb.classes_),
	finalAct="sigmoid")

#opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2 / NUM_EPOCHS)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#model.compile(loss="categorical_crossentropy", optimizer=opt,
#	metrics=["accuracy"])

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training w/ generator...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=NUM_TRAIN_IMAGES // BS,
	validation_data=testGen,
	validation_steps=NUM_TEST_IMAGES // BS,
	epochs=NUM_EPOCHS)

# re-initialize our testing data generator, this time for evaluating
testGen = csv_image_generator(TEST_CSV, BS, mlb,
	mode="eval", aug=None)

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability
predIdxs = model.predict_generator(testGen,
	steps=(NUM_TEST_IMAGES // BS) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print("[INFO] evaluating network...")
print(classification_report(testLabels.argmax(axis=1), predIdxs,
	target_names=mlb.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()


# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
