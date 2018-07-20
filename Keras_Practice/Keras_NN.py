#Purpose: The following code will create a neural network using Keras
#Created by: Nick Kyriacou
#Created on: 7/19/2018

#Importing Packages
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import struct as st
import gzip


# Reads in MNIST dataset
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr


def y_as_matrix(y,training_sets): #This takes a training_setsx1 vector and makes it into a training_sets x num_classes matrix.
	y = np.ravel(y)
	y_array = np.zeros((training_sets,num_classes))
	for i in range(len(y)):
		for j in range(num_classes):
			if (y[i] == j):
				y_array[i][j] = 1
			else:
				y_array[i][j] = 0
	return(y_array)

# Main Code
num_classes = 10
training_images = read_idx('data/train-images-idx3-ubyte.gz',60000)
training_labels = read_idx('data/train-labels-idx1-ubyte.gz',60000)

testing_images = read_idx('data/t10k-images-idx3-ubyte.gz',10000)
testing_labels = read_idx('data/t10k-labels-idx1-ubyte.gz',10000)

training_images = np.reshape(training_images,(60000,784))
testing_images = np.reshape(testing_images,(10000,784))

#Normalize data as well
training_images = training_images/255.0
testing_images = testing_images/255.0

#Take training labels and make it a (60000,10) matrix
training_labels_mat = y_as_matrix(training_labels,60000)
testing_labels_mat = y_as_matrix(testing_labels,10000)

#Now we should create our model in Keras

model = Sequential()

#Keep adding layers based on how we want to structure our NN

model.add(Dense(60000,input_dim = 784,activation = 'sigmoid'))
model.add(Dense(100,activation = 'sigmoid'))
model.add(Dense(100,activation = 'sigmoid'))
model.add(Dense(10,activation = 'softmax'))

#Next compile it

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

model.fit(training_images,training_labels_mat,epochs=200,batch_size = 1000)


