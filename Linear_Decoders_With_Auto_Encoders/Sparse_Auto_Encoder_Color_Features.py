#PURPOSE: The purpose of this function is to implement a Sparse Auto Encoder that will learn features on color images from the STL-10 dataset. These features will later be used in a different exercise on convolution and pooling for classifying STL-10 images. Unlike in the previous two exercises however, our sparse autoencoder will use a linear decoder. The cost and gradient functions will be different to reflect the change from a sigmoid to a linear decoder. 
#CREATED ON: 6/18/2018
#CREATED BY: Nick Kyriacou


########################## IMPORTING PACKAGES############################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
import scipy.io
import matplotlib.image as mpimg
from scipy.optimize import minimize 
import struct as st
import gzip
######################## FUNCTION DEFINITIONS #############################
def create_weights_and_biases(): #This function creates our theta weights to use and has them initialized to within some values [-epsilon,epsilon]
	B_1 = np.random.rand(length_hidden,1)
	B_1 = B_1*2*epsilon - epsilon

	W_1 = np.random.rand(length_hidden,features)
	W_1 = W_1*2*epsilon - epsilon

	B_2 = np.random.rand(features,1)
	B_2 = B_2*2*epsilon - epsilon

	W_2 = np.random.rand(features,length_hidden)
	W_2 = W_2*2*epsilon - epsilon

	theta_1 = np.concatenate((np.ravel(W_1),np.ravel(B_1)))
	theta_2 = np.concatenate((np.ravel(W_2),np.ravel(B_2)))

	theta_all = np.concatenate((np.ravel(theta_1),np.ravel(theta_2)))
	return(theta_all)

def seperate(theta_all): #This function seperates the total theta vector into its individual components
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features))
	B_1 = np.reshape(theta_all[features*length_hidden:features*length_hidden + length_hidden],(length_hidden,1))
	W_2 = np.reshape(theta_all[features*length_hidden+length_hidden:2*features*length_hidden+length_hidden],(features,length_hidden))
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1))
	
	return(W_1,B_1,W_2,B_2)
def linear_hypothesis(theta_2,a_2): # This is the linear hypothesis function for which we calculate the output layer
	#First recover W_2 and B_2
	W_2 = np.reshape(theta_2[0:features*length_hidden],(features,length_hidden))
	B_2 = np.reshape(theta_2[features*length_hidden:len(theta_2)],(features,1))
	
	lin_hypo = np.matmul(W_2,a_2) + B_2
	return(lin_hypo)

def sigmoid(theta_1,a_1,B_1): #This function computes the sigmoid function used to calculate the hidden layer
	z = np.dot(theta_1,a_1.T) + B_1
	sig = 1.0/(1.0+np.exp(-z))
	return(sig)

def feed_forward(theta_all,inputs): #This function feeds forward our inputs by using the sigmoid to find the hidden layer and the linear hypothesis to calculate the output layers
	#First let's reconstruct theta_1 and theta_2
	W_1,B_1,W_2,B_2 = seperate(theta_all)
	theta_1 = np.concatenate((np.ravel(W_1),np.ravel(B_1)))
	theta_2 = np.concatenate((np.ravel(W_2),np.ravel(B_2)))
	#Next let us forward propagate using our two different hypothesis functions
	a_2 = sigmoid(W_1,inputs,B_1)
	
	a_3 = linear_hypothesis(theta_2,a_2)

	return(a_3.T,a_2.T)

def sparse_cost_function_linear_decoder(theta_all,inputs,outputs): # This is the sparse cost function that has been modified to use a linear decoder. 
	a_3, a_2 = feed_forward(theta_all, inputs)
	# Seperate and reshape the Theta values
	W_1, B_1, W_2, B_2 = seperate(theta_all)
	# Finds the average activation of each hidden unit for the entire training set
	p_hat = (1.0 / float(len(inputs)))*np.sum(a_2, axis=0) # 200 len vector (length_hidden,) vector
	diff = a_3 - outputs
	# Calculate Cost as a function of W,B, lambda, and Beta
	Cost_first = (0.5/float(len(inputs)))*np.sum((diff)**2)
	Cost_second = Cost_first + (0.5*l) * (   np.sum(W_1**2)+np.sum(W_2**2)   )
	Cost_third = Cost_second + Beta * np.sum( (  P*np.log(P / p_hat) + (1-P)*np.log((1-P)/(1-p_hat))  ) )
	return Cost_third

def back_prop(theta_all, inputs,outputs): #This back-propagation function calculates the gradient of the cost function for all values of W1, B1, W2, B2
	
	#We need to keep track of total iterations and every 20 times write out our thetas to the output file
	global global_iterations
	global_iterations = global_iterations + 1 
	if (global_iterations%20 == 0):
		np.savetxt(output_name,theta_all,delimiter = ',')
		print('Iteration Number ',global_iterations)

	# Seperate and reshape the W and b values
	W_1, B_1, W_2, B_2 = seperate(theta_all)
	# Forward Propagate
	a_3, a_2 = feed_forward(theta_all, inputs)	# a2 (g.m x 25), a3 (g.m x 64)
	
	# Finds the average activation of each hidden unit for the entire training set
	p_hat = (1.0 / float(len(inputs)))*np.sum(a_2, axis=0) # 200 len vector
	print('in back_prop')
	print(p_hat.shape)
	#Next we must calculate our individual error terms
	delta_3 = a_3 - outputs    #Dimensions of delta_3 are training_sizex192
	# Creating (Capital) Delta matrices
	DeltaW1 = np.zeros(W_1.shape)			# (g.f2, g.f1)
	DeltaW2 = np.zeros(W_2.shape)			# (g.f1, g.f2)
	Deltab1 = np.zeros(B_1.shape)			# (g.f2, 1)
	Deltab2 = np.zeros(B_2.shape)			# (g.f1, 1)
	# Calculate error term for each element in the dataset and add it's contributions to the capital Delta terms

	
	# Calculate Sparsity contribution to delta2
	p_hat = (1.0 / float(len(inputs)))*np.sum(a_2, axis=0)
	K_L = Beta * ( (-P/p_hat) + ((1-P)/(1-p_hat))	)
	something = np.dot(delta_3,W_2)
	next= something+ K_L
	delta_2 = np.multiply( np.dot(delta_3, W_2) + K_L, a_2*( 1.0 -a_2) ) #delta_2 is a training_size x length_hidden dimension matrix
	
	#Next let us compute the partial derivatives
	Delta_W_1 = np.dot(delta_2.T,inputs)			# DIMENSIONS OF (length_hidden, features)
	Delta_W_2 = np.dot(delta_3.T,a_2)		# DIMENSIONS OF (features, length_hidden)
	Delta_B_1 = np.mean(delta_2,axis = 0)		# DIMENSIONS OF (length_hidden, 1)
	Delta_B_2 = np.mean(delta_3,axis = 0)			# DIMENSIONS OF (features, 1)


	#Adds in regularization component of Gradient
	Grad_W_1 = (1.0/float(len(inputs)))*(Delta_W_1) + (l*W_1)
	Grad_W_2 = (1.0/float(len(inputs)))*(Delta_W_2) + (l*W_2)
	
	Grad_B_1 = Delta_B_1
	Grad_B_2 = Delta_B_2

	Combined_Grad = np.concatenate((np.ravel(Grad_W_1),np.ravel(Grad_B_1),np.ravel(Grad_W_2),np.ravel(Grad_B_2)))


	return ( Combined_Grad )


############################ INITIALIZING CONSTANTS AND PREPARING DATA #################



training_size = 100000 #FIX ME
global_iterations = 0 #Number of global iterations in our back_prop and cost function
features = 192 #This number comes from the fact that we have 8x8x3 pixels (RBG)
length_hidden  = 400
l = 3e-3 # Lambda Parameter for regularization
Beta = 5.0 # (Beta) Coefficient of Sparse term (determines relative importance of terms in cost function. (Acts like alpha in that it helps determine size of each step in minimizing the cost function
P = 0.035 #Expected Activation of the Hidden Unit averaged across the training set (Rho). The representation becomes sparser and sparser and Rho decreases
epsilon = 0.12 #Used to create weights within a small range close to zero.
output_name = 'output_folder/optimal_thetas_l_' + str(l)+ '_Beta_' + str(Beta) + '_Rho_' + str(P) + '_.out'
############################# MAIN CODE STARTS HERE ######################################

#First let us randomly initialize a set of weights
theta_all = create_weights_and_biases()
print(theta_all.shape)
W_1,B_1,W_2,B_2 = seperate(theta_all)
print(W_1.shape)
print(B_1.shape)
print(W_2.shape)
print(B_2.shape)

################## NEXT IMPORTANT ADDITION THAT NEEDS TO BE MADE FOR THIS CODE IS TO BE ABLE TO READ INPUTS

data = np.genfromtxt('data/stlSampledPatches_ZCA_Whitened.out',dtype = float)
print(data.shape)
data = np.asarray(data.reshape(training_size,features))
print(data.shape)
y = data #For a sparse auto encoder the outputs = inputs

##### NOW WE CAN BEGIN TESTING FUNCTIONS AND OPTIMIZING WEIGHTS #####
#sample = data[0:500,:] #This selects a small portion of the data to conduct testing on
#y = sample
#First let's check if our cost function works as expected
#print(sample.shape)

print('Initial cost Function')
print(sparse_cost_function_linear_decoder(theta_all,data,y))

#Next we must check our back_prop function
#print('checking back prop')
#print(scipy.optimize.check_grad(sparse_cost_function_linear_decoder,back_prop,theta_all,sample,y))
#Back Propagation gives us 2.6469e-05 which is sufficiently small for us to conclude that back_prop works
#Now that cost function works we can optimize it to find the best weights that minimize the cost function

optimize_time = scipy.optimize.minimize(fun = sparse_cost_function_linear_decoder, x0 = theta_all, method = 'L-BFGS-B', tol = 1e-4, jac = back_prop, args = (data, y)) 
best_thetas = optimize_time.x
 
#Next we take these weights and reapply them to the cost function to see what our minimal cost function is
print('Optimal Cost function is')
best_thetas_cost = sparse_cost_function_linear_decoder(best_thetas,data,y)
print(best_thetas_cost)
#Next let us write out these best thetas to the data file
np.savetxt(output_name, best_thetas,delimiter= ',')

