# import the necessary packages
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class VGG16:
	@staticmethod
	def build(classes):
	    VGG_model = VGG16()
	    #Create new Model with VGG layers
	    model = Sequential()
	    for layer in VGG_model.layers[:-1]:
		model.add(layer)
	    #Freezing all the layers so they won't be trained again
	    #except fc1 and fc2
	    for layer in new_model.layers[:-2]:
		layer.trainable = False
	    # softmax classifier
	    model.add(Dense(classes))
	    model.add(Activation(activation ='sigmoid'))

	    # return the constructed network architecture
	    return model
