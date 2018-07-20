#Purpose: The following code will create a Convolutional Neural Network using Keras
#Created by: Nick Kyriacou
#Created on: 7/19/2018

#Importing Packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import struct as st
import gzip
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input,Convolution2D,MaxPooling2D,Dropout,Activation,Flatten
from keras.utils import np_utils # utiliies for encoding ground truth values

# GLOBAL CONSTANT DEFINITIONS
input_shape = (28,28,1)
conv_dim = 5 #convolving 5x5 feature regions
batch_size = 128
num_iterations = 15
pooling_dim = 2 # using 2x2 pooling regions
testing_size = 10000
training_size = 60000
num_classes = 10
conv_step = 1 #stepping increment for convolution
pool_step = 2 #stepping increment for pooling
output_channel_first_conv = 32
output_channel_second_conv = 64
input_nodes_second_layer = 750

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
training_images = read_idx('data/train-images-idx3-ubyte.gz',training_size)
training_labels = read_idx('data/train-labels-idx1-ubyte.gz',training_size)

testing_images = read_idx('data/t10k-images-idx3-ubyte.gz',testing_size)
testing_labels = read_idx('data/t10k-labels-idx1-ubyte.gz',testing_size)

#For some reason, and I'm not sure why we have to reshape the data into a 4D tensor (sample_size,feature_dim,feature_dim,1)

training_images = np.reshape(training_images,(training_size,28,28,1))
testing_images = np.reshape(testing_images,(testing_size,28,28,1))

#Normalize data as well
training_images = training_images/255.0
testing_images = testing_images/255.0

#Take training labels and make it a (60000,10) matrix
training_labels_mat = y_as_matrix(training_labels,training_size)
testing_labels_mat = y_as_matrix(testing_labels,testing_size)

#Let's specify the input and output of our model in Keras
output = Dense(num_classes,activation = 'softmax')

print('shapes')
print(training_images.shape)

#Now we should create our model in Keras
print('version of keras: ', keras.__version__ )

model = Sequential()

#Keep adding layers based on how we want to structure our CNN

model.add(Convolution2D(output_channel_first_conv,kernel_size = (conv_dim,conv_dim),strides=(conv_step,conv_step),activation = 'relu',input_shape = (28,28,1) ))
model.add(MaxPooling2D(pool_size=(pooling_Dim,pooling_Dim),strides = (pool_step,pool_step)))

#Now let's add another region of convolving and Pooling

model.add(Convolution2D(output_channel_second_conv,kernel_size = (conv_dim,conv_dim),strides = (conv_step,conv_step),activation = 'relu')) 
#Maybe change strides for this convolution if I don't get good results
model.add(MaxPooling2D(pool_size = (pooling_Dim,pooling_Dim),strides = (pool_step,pool_step)))

#Next after convolving we need to flatten the output to enter the fully connected layers
model.add(Flatten())

#Next we can create our neural network. In this instance let's start by having one hidden layer
model.add(Dense(input_nodes_second_layer,activation='relu'))
model.add(Dense(num_classes,activation ='softmax')) #This is our output layer
'''
model.add(Dense(60000,input_dim = 784,activation = 'sigmoid'))
model.add(Dense(100,activation = 'sigmoid'))
model.add(Dense(100,activation = 'sigmoid'))
model.add(Dense(10,activation = 'softmax'))
'''
#Next compile it

model.compile(loss = 'categorical_crossentropy',optimizer = keras.optimizers.SGD(lr=0.01),metrics = ['accuracy'])

model.fit(training_images,training_labels_mat,epochs=num_iterations,batch_size = batch_size,verbose = 1,validation_data = (testing_images,testing_labels_mat)) 
#Next time let's practice adding in a callback feature

#Finally we can evaluate the model
score = model.evaluate(testing_images,testing_labels_mat,verbose = 0)
print('Test loss: ', score[0])
print('Test accuracy: ',score[1])
print(score)

