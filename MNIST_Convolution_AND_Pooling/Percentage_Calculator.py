#Purpose: This code will take the optimal weights calculated using a soft max classifier and use these optimal weights to see how well we can correctly predict the four different classes of images
#Created by: Nick Kyriacou
#Created on: 6/26/2018

################ IMPORTING PACKAGES ##############################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io
from scipy.optimize import minimize 
import scipy.signal
from Convolution import Convolve
from Pooling import Pool
import random
import sys
import struct as st
import gzip



############################# FUNCTION STUFF! ############################

def feed_forward(theta_all,xvals): #This is fine and stays the same
	
	#Let's create all our individual weight terms from theta_all
	W_1,B_1,W_2,B_2 = seperate(theta_all)
	z = np.dot(W_1,xvals.T) + B_1
	a_2 = sigmoid(z)    # (input_size, length_hidden) matrix
	
	# Second run
	a_3 = soft_max_hypo(W_2,B_2,a_2)  # (input_size, num_classes) matrix
	return(a_3,a_2)


def read_ids(filename, n = None): #This function used to read MNIST dataset
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB',f.read(4))
		shape  = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(),dtype = np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
			return arr




def sigmoid(z): 
	hypothesis = 1.0/(1.0 + np.exp(-z) )
	return(hypothesis.T)

def soft_max_hypo(W,B,inputs): #Add surens fix if there are overflow errors

	Max = np.amax(np.matmul(W, inputs.T) + B)
	numerator = np.exp( np.matmul(W, inputs.T) + B - Max )	# 4 x 2000
	denominator = np.asarray([np.sum(numerator, axis=0)])


	return(numerator/denominator).T



#################### FUNCTION DEFINITIONS #########################


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

def seperate(theta_all):
	W_1 = np.reshape(theta_all[0:length_hidden_new*features],(length_hidden_new,features) ) # (25,400)
	B_1 = np.reshape(theta_all[length_hidden_new*features:length_hidden_new*features + length_hidden_new],(length_hidden_new,1) ) # (25,1)
	

	W_2 = np.reshape(theta_all[length_hidden_new*features + length_hidden_new:length_hidden_new*features + length_hidden_new + num_classes*length_hidden_new],(num_classes,length_hidden_new) ) # (10,25)
	B_2 = np.reshape(theta_all[length_hidden_new*features + length_hidden_new + num_classes*length_hidden_new:],(num_classes,1) ) # (10,1)
		
	return(W_1,B_1,W_2,B_2)

############################## CONSTANT DEFINITIONS ########################
features = 400 #Comes from 100*2*2 = 4 * 100 = 400
length_training = 60000 #FIXME This must be changed if the training set has different input size
length_testing = 10000 #FIXME This must be changed if the test set has different input size
length_hidden_initial  = 100 # FIXME This is the number of hidden layers our initial sparse autoencoder used
length_hidden_new = 100 # FIXME This can be changed. It is the number of hidden layers our NN will now use. 
num_classes = 10 #We are classifying images into either dog, cat, plane or car
global_iterations = 0
patch_size = 15 # 8x8
imageDim = 28 #64x64 pixel images
pool_size = 2 #dimension of each pooling region
epsilon = 0.12
l = 1e-4 #FIXME

############################ MAIN CODE STARTS HERE ########################


#First lets grab our test and training images and optimal theta values

optimal_thetas = np.genfromtxt('output_folder/optimal_thetas_Soft_Max_l_'+ str(l) + '.out',dtype = float)


#Now let's grab our MNIST data
training_data = read_ids('data/train-images-idx3-ubyte.gz',length_training)
training_labels = read_ids('data/train-labels-idx1-ubyte.gz',length_training)
testing_data = read_ids('data/t10k-images-idx3-ubyte.gz',length_testing)
testing_labels = read_ids('data/t10k-labels-idx1-ubyte.gz',length_testing)

#Let's also normalize our testing and training data
training_data = training_data/255.0
testing_data = testing_data/255.0

#Let's turn our test labels into a matrix
test_labels_mat = y_as_matrix(testing_labels,length_testing)


file_name = 'output_folder/MNISTTest_Set_CONVED_AND_POOLED_STEP_25_PoolDim_710_B_0.5_Rho_0.05.out'
input_features_testing = np.genfromtxt(file_name, dtype = float)



input_features_testing = np.reshape(input_features_testing,(length_hidden_initial,length_testing,pool_size,pool_size)) #DIMENSIONS OF (100,10000,2,2)
#However we need to keep reshaping this because our input data needs to be a 2D matrix for soft_max classifier to work

input_features_testing = np.swapaxes(input_features_testing,0,1) #Now the input is a (10000,100,2,2)
print(input_features_testing.shape)
#Making this a 2D array gives us
input_features_testing = np.reshape(input_features_testing,(length_testing, (pool_size**2)*length_hidden_initial)) #Now our input is a (10000,400)
 


#Now let us feed forward these inputs to get our output and hidden layer
a_3,a_2 = feed_forward(optimal_thetas,input_features_testing)



#NOW LET'S CALCULATE THE PERCENTAGES EACH CLASS WAS GUESSED CORRECTLY

#First let's find the total number of each type of image within the test_set
class_count = np.zeros((num_classes))
print(class_count.shape)
for i in range(length_testing):
	for j in range(num_classes):
		if (testing_labels[i] == j):
			class_count[j] = class_count[j] + 1.0

#Next let's see whether or not we guessed the images correctly

guess_for_test_data = np.zeros((length_testing))
print(guess_for_test_data.shape)
for i in range(length_testing):
	guess_for_test_data[i] = (np.argmax(a_3[i]))

print(guess_for_test_data)

correctly_guessed_digit = np.zeros((num_classes))
for i in range(length_testing):
	if (testing_labels[i] == guess_for_test_data[i]):
		digit = testing_labels[i]
		correctly_guessed_digit[digit] = correctly_guessed_digit[digit] + 1.0

#Next let's calculate the percentage
print('Percentages that each image class was correctly guessed: ')
percentages = (correctly_guessed_digit/class_count)*100.0
print(percentages[0])
print(percentages[1])
print(percentages[2])
print(percentages[3])
print(percentages[4])
print(percentages[5])
print(percentages[6])
print(percentages[7])
print(percentages[8])
print(percentages[9])



