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
from Create_test_and_train_images import generate_images
from Convolution import Convolve
from Pooling import Pool
import random
import sys




############################# FUNCTION STUFF! ############################

def feed_forward(theta_all,xvals): #This is fine and stays the same
	
	#Let's create all our individual weight terms from theta_all
	W_1,B_1,W_2,B_2 = seperate(theta_all)
	z = np.dot(W_1,xvals.T) + B_1
	a_2 = sigmoid(z)    # (input_size, length_hidden) matrix
	
	# Second run
	a_3 = soft_max_hypo(W_2,B_2,a_2)  # (input_size, num_classes) matrix
	return(a_3,a_2)




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
	W_1 = np.reshape(theta_all[0:length_hidden*features],(length_hidden,features) ) # (25,3600)
	B_1 = np.reshape(theta_all[length_hidden*features:length_hidden*features + length_hidden],(length_hidden,1) ) # (25,1)
	

	W_2 = np.reshape(theta_all[length_hidden*features + length_hidden:length_hidden*features + length_hidden + num_classes*length_hidden],(num_classes,length_hidden) ) # (4,25)
	B_2 = np.reshape(theta_all[length_hidden*features + length_hidden + num_classes*length_hidden:],(num_classes,1) ) # (4,1)
		
	return(W_1,B_1,W_2,B_2)

############################## CONSTANT DEFINITIONS ########################

features = 3600 #Comes from 400*3*3 = 1200 * 3 = 3600
features_1 = 400 #This is used to help reshape the data into a (400,input_size,3,3)
length_training = 2000 #FIXME This must be changed if the training set has different input size
length_test = 3200 #FIXME This must be changed if the test set has different input size
length_hidden  = 25 # FIXME This can be chosen depending on how many hidden layer nodes we want for our NN
num_classes = 4 #We are classifying images into either dog, cat, plane or car
pool_size = 3 #dimension of each pooling region
global_iterations = 0

############################ MAIN CODE STARTS HERE ########################


#First lets grab our test and training images and optimal theta values

optimal_thetas = np.genfromtxt('output_folder/optimal_thetas_Soft_Max.out',dtype = float)
training_images,training_labels,test_images,test_labels = generate_images()

num_test_images = len(test_images[0,0,0,:])
test_labels = test_labels - 1 #Formats range from [1,4] to [0,3] 
test_labels_mat = y_as_matrix(test_labels,num_test_images)


file_name = 'output_folder/Test_Set_CONVED_AND_POOLED_STEP_25.out'
input_features_testing = np.genfromtxt(file_name, dtype = float)



input_features_testing = np.reshape(input_features_testing,(features_1,num_test_images,pool_size,pool_size)) #DIMENSIONS OF (400,2000,3,3)
#However we need to keep reshaping this because our input data needs to be a 2D matrix for soft_max classifier to work

input_features_testing = np.swapaxes(input_features_testing,0,1) #Now the input is a (2000,400,3,3)
#Making this a 2D array gives us
input_features_testing = np.reshape(input_features_testing,(num_test_images, (pool_size**2)*features_1))



#Now let us feed forward these inputs to get our output and hidden layer
a_3,a_2 = feed_forward(optimal_thetas,input_features_testing)



#NOW LET'S CALCULATE THE PERCENTAGES EACH CLASS WAS GUESSED CORRECTLY

#First let's find the total number of each type of image within the test_set
class_count = np.zeros((num_classes))
print(class_count.shape)
for i in range(num_test_images):
	for j in range(num_classes):
		if (test_labels[i] == j):
			class_count[j] = class_count[j] + 1.0

#Next let's see whether or not we guessed the images correctly

guess_for_test_data = np.zeros((num_test_images))
print(guess_for_test_data.shape)
for i in range(num_test_images):
	guess_for_test_data[i] = float(np.argmax(a_3[i]))

print(guess_for_test_data)

correctly_guessed_digit = np.zeros((num_classes))
for i in range(num_test_images):
	if (test_labels[i] == guess_for_test_data[i]):
		digit = test_labels[i]
		correctly_guessed_digit[digit] = correctly_guessed_digit[digit] + 1.0

#Next let's calculate the percentage
print('Percentages that each image class was correctly guessed: ')
percentages = (correctly_guessed_digit/class_count)*100.0
print(percentages[0])
print(percentages[1])
print(percentages[2])
print(percentages[3])



