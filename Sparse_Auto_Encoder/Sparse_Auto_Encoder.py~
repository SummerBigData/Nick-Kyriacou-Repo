#Here we will build and develop a Sparse Auto Encoder
#Owner: Nick Kyriacou
#Date Created: 6/7/2018

# IMPORTING PACKAGES


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


#DEFINING GLOBAL VARIABLES


global training_sets #5000 training samples
global features #400 features per sample (excluding x-int bias we add)
global l #Lambda constant 
global num_classes #Ten total classes
global iterations #Counts number of iterations for any specific process
global length_hidden #Length of hidden layer (without bias added)
global epsilon #Used in gradient check function
global test_set_size #Size of our test set
global Beta #Weight of Sparsity parameter
global P #Sparsity Parameter

#FUNCTION DEFINITIONS

'''
def read_ids(filename, n = None): #This function used to read MNIST dataset
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB',f.read(4))
		shape  = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(),dtype = np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
			return arr




def grad_sigmoid(z):
	hypo = sigmoid(z)
	grad = hypo * (1.0 - hypo)
	return(grad) 



	
	
def grad_check(theta_all,inputs,outputs):
	theta_upper = theta_all + epsilon
	theta1 = theta_all[0:(features+1)*length_hidden]
	theta2 = theta_all[(features+1)*length_hidden:len(theta_all)]
	theta1 = np.reshape(theta1,(length_hidden,features+1))
	theta2 = np.reshape(theta2,(num_classes,length_hidden+1))
	#Now re-roll them back up
	theta_unrolled_upper= np.concatenate((np.ravel(theta1),np.ravel(theta2)))
	cost_function_upper = cost_function_reg(theta_unrolled_upper,inputs,outputs)
	theta_lower = theta_all - epsilon
	theta1 = theta_all[0:(features+1)*length_hidden]
	theta2 = theta_all[(features+1)*length_hidden:len(theta_all)]
	theta1 = np.reshape(theta1,(length_hidden,features+1))
	theta2 = np.reshape(theta2,(num_classes,length_hidden+1))
	#Now re-roll them back up
	theta_unrolled_lower= np.concatenate((np.ravel(theta1),np.ravel(theta2)))
	cost_function_lower = cost_function_reg(theta_unrolled_lower,inputs,outputs)
	grad_check = (cost_function_upper - cost_function_lower)/float(2*epsilon)
	return(grad_check)


def cost_function_reg(theta_all,inputs,outputs):
	#First we must unravel our theta1 and theta2 and reshape them into the correct dimensions
	theta1 = theta_all[0:(features+1)*length_hidden]
	theta2 = theta_all[(features+1)*length_hidden:len(theta_all)]
	theta1 = np.reshape(theta1,(length_hidden,features+1))
	theta2 = np.reshape(theta2,(num_classes,length_hidden+1))
	hypothesis,a_2_bias = feed_forward(theta_all,inputs)
	

	J = (1.0/float(training_sets))*np.sum(   -1*np.multiply(outputs, np.log(hypothesis)) - np.multiply(1-outputs, np.log(1 - hypothesis)	) )
	J = J + (0.5 * l / float(training_sets)) * ( np.sum(theta1**2) - np.sum(theta1[0]**2) + np.sum(theta2**2) )
	#J = J + (0.5 * l/float(training_sets))*(   np.sum(theta1**2) - np.sum(theta1[0]**2) + np.sum(theta2**2) - np.sum(theta2.T[0]**2)	)
	#print('did this:cost_func_reg: ', J)
	return(J)


def cost_function(theta_all,inputs,outputs):
	#print('inside cost function')
	#print(theta_all.shape)
	theta1 = theta_all[0:(features+1)*length_hidden]
	theta2 = theta_all[(features+1)*length_hidden:len(theta_all)]
	theta1 = np.reshape(theta1,(length_hidden,features+1))
	theta2 = np.reshape(theta2,(num_classes,length_hidden+1))
	hypothesis,a_2_bias = feed_forward(theta_all,inputs)
	first = -1.0*np.multiply(outputs,np.log(hypothesis))
	second = np.multiply((1.0 - outputs), np.log(1 - hypothesis))
	cost_function = (1.0/float(training_sets))*np.sum((first - second))
	#print('did this: cost_func',cost_function)
	return(cost_function)
	



def y_as_matrix(y,training_sets): #This takes a 5000x1 vector and makes it into a 5000x10 matrix, Code based on one made by suren
	y = np.ravel(y)
	y_array = np.zeros((training_sets,10))
	for i in range(len(y)):
		for j in range(10):
			if (y[i] == j):
				y_array[i][j] = 1
	return(y_array)


'''


def create_weights_and_bias(): #This function creates all W's and B's and combines them into an unrolled list and returns said list
	#It is important that all weights are randomly created on an interval [-epsilon,epsilon] very close to zero
	theta1_random = np.random.rand(length_hidden,features) #25x64 matrix
	theta1_random = theta1_random*2*epsilon - epsilon
	theta2_random = np.random.rand(features,length_hidden) #64x25 matrix
	theta2_random = theta2_random*2*epsilon - epsilon
	
	bias_1 = np.random.rand(length_hidden,1) #25x1 matrix
	bias_1 = bias_1*2*epsilon - epsilon
	bias_2 = np.random.rand(features,1) #64x1 matrix
	bias_2 = bias_2*2*epsilon - epsilon
	theta_all = np.concatenate((np.ravel(theta1_random), np.ravel(theta2_random), np.ravel(bias_1),np.ravel(bias_2)))
	print('in creating weights')
	print(theta_all.shape)
	return(theta_all)
	
def seperate(theta_all): #This function will take a combined theta vector and seperate it into 4 of its specific components
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features))
	B_1 = np.reshape(theta_all[2*features*length_hidden:2*features*length_hidden + length_hidden],(length_hidden,1))
	W_2 = np.reshape(theta_all[features*length_hidden:2*features*length_hidden],(features,length_hidden))
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1))
	#Now that we have seperated each vector and reshaped it into usable format lets return each weight
	print('in seperate')
	print(W_1.shape)
	print(W_2.shape)
	print(B_1.shape)
	print(B_2.shape)
	return(W_1,B_1,W_2,B_2)


def feed_forward(theta_all,xvals):
	#First we must unravel our theta1 and theta2 and reshape them into the correct dimensions
	W_1,B_1, W_2, B_2 = seperate(theta_all)
	print('inside feed forward')
	print(W_1.shape)
	print(B_1.shape)
	print(xvals.shape)
	a_2 = sigmoid(W_1,B_1,xvals) #Calculates a_2 as a matrix
	print(a_2.shape)
	print('that was a_2')
	a_3 = sigmoid(W_2,B_2,a_2)
	print(a_3.shape)
	#print('did this:feed_forward')
	return(a_3,a_2)


def sigmoid(W,B,inputs): #Add surens fix if there are overflow errors
	z = np.matmul(W, inputs.T) + B #Should be a 25x10000 matrix
	print('in sigmoid')
	print(z.shape)
	hypo = 1.0/(1.0 + np.exp(-1.0*z))
	return(hypo)

def sparse_cost_function(theta_all,inputs):
	W_1,B_1, W_2, B_2 = seperate(theta_all)
	print('in cost function')
	print(W_1.shape)
	print(W_2.shape)
	print(B_1.shape)
	print(B_2.shape)
	#First forward propogate
	a_3,a_2 = feed_forward(theta_all,inputs)
	#Now find sum of squared difference of inputs  - outputs
	diff = inputs - a_3
	print(diff.shape)
	sum_diff_squared = ( 1/(2*float(training_sets)) ) * np.sum(np.multiply(diff,diff))
	print(sum_diff_squared)
	J = ( 0.5*float(l) ) * ( np.sum(np.multiply(W_1,W_1)) + np.sum(np.multiply(W_2,W_2)) )
	print(J)
	J = J + sum_diff_squared
	#Next we need to calculate the sparsity parameter
	p_hat = (1.0/float(len(inputs)))* np.sum(a_2,axis = 0)
	print(p_hat)
	J = J/float(len(inputs)) + Beta * np.sum( P*np.log(P/p_hat) + (1.0 - P)*(np.log((1.0 - P)/(1.0 - p_hat)) ) ) 
	return(J)

def back_prop(theta_all,inputs):
	print('Im in backprop')
	W_1,B_1, W_2, B_2 = seperate(theta_all)
	#First Forward Propogate
	a_3, a_2 = feed_forward(theta_all, inputs)
	print('a_2 looks like', a_2.shape)
	p_hat = (1.0/float(len(inputs)))* np.sum(a_2,axis = 0)
	K_L =  Beta*(-1.0*(P/p_hat) + (1.0 - P)/(1.0 - p_hat))
	K_L = K_L.reshape(1,length_hidden)
	#Now let's find errors in each layer including sparsity contribution to Delta2
	Delta3 = np.multiply( (a_3 - inputs),a_3*(1.0-a_3) )
	Delta2 = np.multiply( np.dot(Delta3.T,W_2.T) + K_L, a_2*(1.0-a_2) )

	#Next we must compute the partial derivatives
	grad_W1 = np.dot(Delta2,inputs.T)
	grad_W2 = np.dot(Delta3,a_2.T)
	grad_B1 = np.sum(Delta2,axis=1)/float(len(inputs))
	grad_B2 = np.sum(Delta3,axis=1)/float(len(inputs))


	#Now adding in regularization component
	grad_W1 = (grad_W1 + l*W_1)/float(len(inputs)) 
	grad_W2 = (grad_W2 + l*W_2)/float(len(inputs)) 
	Combined_Grad = np.concatenate(np.ravel(grad_W1),np.ravel(grad_W2),np.ravel(grad_B1),np.ravel(grad_B2))
	print('in backprop')
	print(combined_Grad.shape)
	return(combined_Grad)


#MAIN SECTION OF CODE STARTS HERE


l = 1 #Lambda constant 
length_hidden = 25 #Length of hidden layer (without bias added)
epsilon = .12 #Used in gradient check function
training_sets = 10000 #10000 samples
features = 64 #64 features comes from 8x8 hidden layer
Beta = 0 #Weight of Sparsity Parameter
P = 0.05 #Sparsity Parameter


#For a sparse auto encoder our x = y
data = np.genfromtxt('output_folder/10000Random8x8.out')
print(data.shape)
#Now we need to reshape the data into a 64x10000 matrix
data = np.reshape(data,(64,10000))
print(data.shape)
y_vals = data #outputs = inputs
y_vals = y_vals.T
print('yvals looks like' , y_vals.shape)

#Now let's grab our weights and bias terms
theta_all = create_weights_and_bias()
W_1,B_1, W_2, B_2 = seperate(theta_all)

print( ' looking at backprop')
print(back_prop(theta_all,y_vals))

cost = sparse_cost_function(theta_all,y_vals)
print('original cost is ' + str(cost) )

print(theta_all.shape)
#Now let's minimize our cost function
print('Lambda is ', l)
print('now we are optimizing the cost function')
optimize_time = scipy.optimize.minimize(fun = sparse_cost_function,x0 = theta_all,method = 'CG',tol = 1e-4,jac = back_prop,args = (y_vals))
optimal_thetas = optimize_time.x
optimal_thetas_cost = sparse_cost_function(optimal_thetas,y_vals)
