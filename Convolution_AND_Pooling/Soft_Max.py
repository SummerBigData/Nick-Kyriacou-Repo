#Purpose: This code takes the set of convolved and pooled features from the training set and trains a softmax classifier to map the pooled features to the class labels. Then it will output the trained weights to an output file to be used later to test the softmax classifier on the test set.
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

#################### FUNCTION DEFINITIONS #########################


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

# Calculate the regularized Cost J(theta)
def soft_max_regularized_cost_function(theta_all, inputs,outputs):
	
	# Forward Propagates to output layer
	a_3, a_2 = feed_forward(theta_all, inputs)
	# Seperate and reshape the Theta values
	W_1, B_1, W_2, B_2 = seperate(theta_all)
	# Calculate Sparsity contribution. Hehe, phat sounds like fat (stands for p hat)
	Cost_first = (-1.0 / float(len(inputs)))*np.sum( np.multiply(np.log(a_3), outputs))
	Cost_second = (l / (2.0)) * ( np.sum(W_2**2) + np.sum(W_1**2) )
	Cost_total = Cost_first + Cost_second
	return Cost_total

# Calculate the gradient of cost function for all values of W1, W2, b1, and b2
def back_prop(theta_all, inputs,outputs):
	
	#We need to keep track of total iterations and every 20 times write out our thetas to the output file
	global global_iterations
	global_iterations = global_iterations + 1 
	if (global_iterations%20 == 0):
		#np.savetxt(output_name,theta_2,delimiter = ',') ######Fix me!
		print('Iteration Number ',global_iterations)

	# Seperate and reshape the W and b values
	W_1, B_1, W_2, B_2 = seperate(theta_all)
	# Forward Propagate and create an array of ones to attach onto a_2, and ont

	a_3, a_2 = feed_forward(theta_all, inputs)	# a2 (input_size x 25), a3 (input_size x 4)


	ones_a_2 = np.ones((len(a_2),1)) #DIMENSIONS OF: (input_size,1)
	ones_inputs= np.ones((len(inputs),1)) #DIMENSIONS OF: (input_size,1)


	a_2_int = np.hstack((ones_a_2,a_2)) #DIMENSIONS OF: (input_size,26) =  (input_size,length_hidden+1)

	inputs_int = np.hstack((ones_inputs,inputs)) #DIMENSIONS OF: (98,3601) = (input_size,features+1)


	# We also want to stack together our W_2 and B_2 and W_1 and B_1
	W_B_1_combined = np.hstack((B_1,W_1)) #DIMENSIONS OF: (25,3601) = (length_hidden,features+1)

	W_B_2_combined = np.hstack((B_2,W_2)) #DIMENSIONS OF: (4,26) = (num_classes,length_hidden+1)

 	#Next we need to calculate the error terms from the output to hidden layer (delta_3) and from the hidden to input layer (delta_2)
	delta_3 = outputs - a_3 #DIMENSIONS OF: (input_size,num_classes) = (98,4)
	
	derivative = np.multiply(a_2,(1.0 - a_2) ) #DIMENSIONS OF: (98,25) = (input_size,length_hidden)
	weighted_error = np.matmul(delta_3,W_2)
	delta_2 = np.multiply( weighted_error,derivative) #DIMENSIONS OF: (98,25) = (input_size,length_hidden)
	

	#Next we need to compute the partial derivatives 
	

	Grad_W_2 = np.matmul( (outputs - a_3).T, a_2_int ) #DIMENSIONS OF: (4 , 26) = (num_classes,length_hidden+1)
	Grad_W_1 = np.matmul(delta_2.T,inputs_int) #DIMENSIONS OF: ( 25,3601 ) = (length_hidden,features+1)


	gradient_W_2 = ( -1.0/float(len(inputs)) ) * Grad_W_2 + l*W_B_2_combined #DIMENSIONS OF:  (4,26) = (num_classes, length_hidden+1)
	gradient_W_1 = ( -1.0/float(len(inputs)) ) * Grad_W_1 + l*W_B_1_combined #DIMENSIONS OF:  (25,3601) = (length_hidden,features+1)

	
	
	# Next let's make this gradient a list (1-D vector) so that we can roll them up and pass back from the back_prop function

	Delta_B_2 = np.ravel(gradient_W_2[:,:1])
	Delta_W_2 = np.ravel(gradient_W_2[:,1:])
	Delta_W_1 = np.ravel(gradient_W_1[:,1:]) 
	Delta_B_1 = np.ravel(gradient_W_1[:,:1]) 

	Grad_theta_all = np.concatenate((Delta_W_1,Delta_B_1,Delta_W_2,Delta_B_2))
	
	return(Grad_theta_all)



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

def create_weights_and_bias():

	W_1 = np.random.rand(length_hidden,features) #25x3600 matrix
	W_1 = W_1*2*epsilon - epsilon
	W_2 = np.random.rand(num_classes,length_hidden) #4x25 matrix
	W_2 = W_2*2*epsilon - epsilon

	B_1 = np.random.rand(length_hidden,1) #25x1 matrix
	B_1 = B_1*2*epsilon - epsilon
	B_2 = np.random.rand(num_classes,1) #4x1 matrix
	B_2 = B_2*2*epsilon - epsilon

	return(W_1,B_1,W_2,B_2)

def seperate(theta_all):
	W_1 = np.reshape(theta_all[0:length_hidden*features],(length_hidden,features) ) # (25,3600)
	B_1 = np.reshape(theta_all[length_hidden*features:length_hidden*features + length_hidden],(length_hidden,1) ) # (25,1)
	

	W_2 = np.reshape(theta_all[length_hidden*features + length_hidden:length_hidden*features + length_hidden + num_classes*length_hidden],(num_classes,length_hidden) ) # (4,25)
	B_2 = np.reshape(theta_all[length_hidden*features + length_hidden + num_classes*length_hidden:],(num_classes,1) ) # (4,1)
		
	return(W_1,B_1,W_2,B_2)
############################ GLOBAL VARIABLE DEFINITIONS ##########################
features = 3600 #Comes from 400*3*3 = 1200 * 3 = 3600
features_1 = 400 #This is used to help reshape the data into a (400,input_size,3,3)
length_training = 2000 #FIXME This must be changed if the training set has different input size
length_test = 3200 #FIXME This must be changed if the test set has different input size
length_hidden  = 25 # FIXME This can be chosen depending on how many hidden layer nodes we want for our NN
num_classes = 4 #We are classifying images into either dog, cat, plane or car
global_iterations = 0
patch_size = 8 # 8x8
num_colors = 3 # (RGB)
imageDim = 64 #64x64 pixel images
pool_size = 3 #dimension of each pooling region
epsilon = 0.12
l = 1e-4 #FIXME

STEP_SIZE = 25 # Convolve and Pool only 50 features at a time to not run out of memory
############################### MAIN CODE STARTS HERE ############################


#First let's import the pooled features for our training set here
file_name = 'output_folder/Train_Set_CONVED_AND_POOLED_STEP_25.out'
input_features_training = np.genfromtxt(file_name, dtype = float)

training_images,training_labels,test_images,test_labels = generate_images()

#Next we will need to reshape this list back into its original array 
num_training_images = len(training_images[0,0,0,:])
input_features_training = np.reshape(input_features_training,(features_1,num_training_images,pool_size,pool_size)) #DIMENSIONS OF (400,2000,3,3)
#However we need to keep reshaping this because our input data needs to be a 2D matrix for soft_max classifier to work

input_features_training = np.swapaxes(input_features_training,0,1) #Now the input is a (2000,400,3,3)
#Making this a 2D array gives us
input_features_training = np.reshape(input_features_training,(num_training_images, (pool_size**2)*features_1)) #Makes a (2000,3600) 3600 comes from 3*3*400 = 9*400 = 3600

#Let's turn the training_labels into an array so that we can run it through our soft max classifier
size_training_labels = len(training_labels)
#We need to manually adjust the range of the training labels originally [1:4], now [0,3]
training_labels = np.ravel(training_labels[:size_training_labels,0])-1
training_labels_mat = y_as_matrix(training_labels,size_training_labels)

#Next lets get our weights up and runnning
W_1,B_1,W_2,B_2 = create_weights_and_bias()

#Let's combine these into theta_1 and theta_2 and ultimately theta_all
theta_1 = np.concatenate((np.ravel(W_1),np.ravel(B_1)))
theta_2 = np.concatenate((np.ravel(W_2),np.ravel(B_2)))
theta_all = np.concatenate((np.ravel(theta_1),np.ravel(theta_2)))
# Just checking whether or not the seperate function works
# W_1,B_1,W_2,B_2 = seperate(theta_all) 

print('Initial Cost Function is: ')
print( soft_max_regularized_cost_function(theta_all, input_features_training,training_labels_mat) )

#Now that we have verified cost function works lets proceed to check our gradient!
'''
sample = input_features_training[0:98,:]
y = training_labels_mat[0:98,:]
print('checking gradient')
print(scipy.optimize.check_grad(soft_max_regularized_cost_function,back_prop,theta_all,sample,y))
'''
#gradient checking gives us 3.4011e-05 THUS IT IS GOOD!!!

print('Now lets optimize our cost function')
optimize_time = minimize(fun=soft_max_regularized_cost_function, x0= theta_all, method='L-BFGS-B', tol=1e-4, jac=back_prop, args=(input_features_training, training_labels_mat) ) # options = {'disp':True}
best_thetas = optimize_time.x
print('optimal cost function is')
print(soft_max_regularized_cost_function(best_thetas,input_features_training,training_labels_mat))

#Next let's write out these optimal thetas to a different file
output_name = 'output_folder/optimal_thetas_Soft_Max.out'
np.savetxt(output_name, best_thetas,delimiter= ',')

