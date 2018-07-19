#Purpose: This code will take the 10000 randomly sampled 15x15 patches from the MNIST data and run it through a sparse autoencoder using the assorted values (a hidden layer size of 100, lambda = 10, Beta = 0.3, Rho = 0.05 because these earlier produced the best penstroke visualizations) to obtain features for this data set
#WEIGHTS ARE STORED AS SUCH (W1,W2,B1,B2)!!!!! KEEP CONSISTENT !!!!!
#Created by: Nick Kyriacou
#Created on: 6/27/2018

########################## IMPORTING PACKAGES #######################

import matplotlib
matplotlib.use('agg')
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


########################### FUNCTION DEFINITIONS #####################
def create_weights_and_bias():
	#It is important that all weights are randomly created on an interval [-epsilon,epsilon] very close to zero
	theta1_random = np.random.rand(length_hidden,features) #100x225 matrix
	theta1_random = theta1_random*2*epsilon - epsilon
	theta2_random = np.random.rand(features,length_hidden) #225x100 matrix
	theta2_random = theta2_random*2*epsilon - epsilon
	
	bias_1 = np.random.rand(length_hidden,1) #100x1 matrix
	bias_1 = bias_1*2*epsilon - epsilon
	bias_2 = np.random.rand(features,1) #225x1 matrix
	bias_2 = bias_2*2*epsilon - epsilon
	theta_all = np.concatenate((np.ravel(theta1_random), np.ravel(theta2_random), np.ravel(bias_1),np.ravel(bias_2)))
	#print('in creating weights')
	#print(theta_all.shape)
	return(theta_all)

def Lin4(a, b, c, d):
	return np.concatenate((np.ravel(a), np.ravel(b), np.ravel(c), np.ravel(d)))

def seperate(theta_all): #This function will take a combined theta vector and seperate it into 4 of its specific components
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features))
	B_1 = np.reshape(theta_all[2*features*length_hidden:2*features*length_hidden + length_hidden],(length_hidden,1))
	W_2 = np.reshape(theta_all[features*length_hidden:2*features*length_hidden],(features,length_hidden))
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1))

	return(W_1, W_2, B_1, B_2)


def feed_forward(theta_all,xvals):
	W_1, W_2, B_1, B_2 = seperate(theta_all)
	# Calculate a2 (g.m x 25)
	a_2 = sigmoid(W_1, B_1, xvals)
	# Calculate and return the output from a2 and W2 (g.m x 64)
	a_3 = sigmoid(W_2, B_2, a_2)
	return a_2, a_3


def sigmoid(W,B,inputs): #Add surens fix if there are overflow errors
	
	z = np.matmul(W, inputs.T) + B 
	#print(z.shape)
	z = np.array(z,dtype = np.float128)
	hypo = 1.0/(1.0 + np.exp(-1.0*z))
	hypo = np.array(hypo.T,dtype = np.float64)
	return(hypo)

# Calculate the regularized Cost J(theta)
def sparse_cost_function(theta_all, inputs):
	# FIrst you gotta propogate all the way to output layer
	a_2, a_3 = feed_forward(theta_all, inputs)
	

	#Next you want to pull out individual theta matrices that will be used later
	W_1, W_2, B_1, B_2 = seperate(theta_all)
	# This is our sparsity term for our cost function
	phat = (1.0 /len(inputs))*np.sum(a_2, axis=0) # 25 len vector
	# Cost function can be split into three successive calculations
	first = (0.5/len(inputs))*np.sum((a_3 - inputs)**2)
	second = (0.5/len(inputs))* l * (np.sum(W_1**2)+np.sum(W_2**2))
	third =  Beta * np.sum(   P*np.log(P / phat) + (1-P)*np.log((1-P)/(1-phat))  )
	cost = first + second + third
	return cost

# Calculates the cost function gradient for each W and B value
def back_prop(theta_all, inputs):
	# Seperate and reshape the W and b values
	W_1, W_2, B_1, B_2 = seperate(theta_all)
	# Forward Propagate
	a_2, a_3 = feed_forward(theta_all, inputs)	# a2 (g.m x 25), a3 (g.m x 64)
	
	# Calculate (Lowercase) deltas for each element in the dataset and add it's contributions to the Deltas
	error = a_3 - inputs
	delta_3 = np.multiply( error, a_3*(1-a_3) )
	# Calculate Sparsity contribution to delta2
	p_hat = (1.0 / len(inputs))*np.sum(a_2, axis=0)
	KL_Divergence = Beta * ( -P/p_hat + (1-P)/(1-p_hat)	)
	#Hidden layer error term broken into several different steps to make troubleshooting easier
	first = np.dot(delta_3,W_2)
	second = first + KL_Divergence.reshape(1, length_hidden)
	delta_2 = np.multiply( second,a_2*(1-a_2) )	
	#delta2 = np.multiply( np.matmul(delta_3, W2) + sparsity.reshape(1, length_hidden), a2*(1-a2) )

	Grad_W_1 = np.dot(delta_2.T, inputs) 	# (25, 64)
	Grad_W_2 = np.dot(delta_3.T, a_2)     	# (64, 25)
	Grad_B_1 = np.mean(delta_2, axis = 0) # (25,) vector
	Grad_B_2 = np.mean(delta_3, axis = 0) # (64,) vector

	return Lin4( (1.0/len(inputs))*(Grad_W_1 + l*W_1) , (1.0/len(inputs))*(Grad_W_2 + l*W_2) , Grad_B_1 , Grad_B_2 )




########################### CONSTANT DEFINITIONS ###########################
input_size = 10000
input_feature_dim = 15

l = 10 #Lambda constant 
length_hidden = 100 #Length of hidden layer (without bias added)
epsilon = .12 #Used in to initialize theta weights function
features = 225 #225 features comes from 15x15 input images layer
Beta = float(0.5) #Weight of Sparsity Parameter
P = float(0.05) #Sparsity Parameter

########################### MAIN CODE STARTS HERE ###########################

#First let's get our input data in
input_name = 'output_folder/randomly_sampled_10k_15x15_pixel_images.out'
data = np.genfromtxt(input_name,dtype = float) #Remember that this input data has already been normalized (divided by 255.0)

data = np.reshape(data,(input_size,input_feature_dim**2))
#data = data/255.0
print(data.shape)

#Next let's make our initial theta weights
theta_all = create_weights_and_bias()

#First let's check to see if our cost function works
print('Initial Cost Function is: ')
print(sparse_cost_function(theta_all,data))


'''
#Next let's check our gradient
sample = data[0:98,:]
print('Checking BackPropagation Algorithm')
print(scipy.optimize.check_grad(sparse_cost_function,back_prop,theta_all,sample))
#Grad lets ~2.65e-6 so this is close enough to being good. We can proceed and decide if there are problems later
'''


#Now let's go about optimizing our cost function
print('Optimizing Cost Function')
optimize_time = scipy.optimize.minimize(fun = sparse_cost_function,x0 = theta_all,method = 'L-BFGS-B',tol = 1e-4,jac = back_prop,args = (data,))
print('optimal cost function is')
optimal_thetas = optimize_time.x
optimal_thetas_cost = sparse_cost_function(optimal_thetas,data)
print(optimal_thetas_cost)

#Now let's save these theta values out to a different value to be used in the future
output_name = 'output_folder/optimal_thetas_l_' + str(l) + '_B_' + str(Beta) + '_Rho_' + str(P) + '.out'
np.savetxt(output_name,optimal_thetas,delimiter = ',')
