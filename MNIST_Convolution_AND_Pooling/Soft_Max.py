#Purpose: This code takes the set of convolved and pooled features from the training set and trains a softmax classifier to map the pooled features to the class labels. Then it will output the trained weights to an output file to store. Finally it will determine how well the softmax classifier performed. 
#Created by: Nick Kyriacou
#Created on: 6/26/2018

################################# IMPORTING PACKAGES #########################
import struct as st
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random


###################### CONSTANT VARIABLE DEFINITIONS  #########################################
training_size = 60000
testing_size = 10000
global_step = 0
image_size = 28
features = 784
pool_Dim = 2
features_pooled = 400   
length_hidden = 36
length_hidden_old = 100
l = 1e-4
num_classes = 10    



################### FUNCTION DEFINITIONS ##################

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

# Change our weights and bias terms back into their proper shapes
def seperate(theta):
	B_1 = np.reshape(theta[features_pooled * length_hidden + length_hidden * num_classes: 
		features_pooled * length_hidden + length_hidden * num_classes + length_hidden ],
		(length_hidden, 1))
	W_1 = np.reshape(theta[0:length_hidden * features_pooled], (length_hidden, features_pooled))
	
	B_2 = np.reshape(theta[features_pooled * length_hidden + length_hidden * num_classes + length_hidden:
		 len(theta)], (num_classes, 1))
	
	W_2 = np.reshape(theta[length_hidden * features_pooled: 
		features_pooled * length_hidden + length_hidden * num_classes], (num_classes, length_hidden))

	return W_1, W_2, B_1, B_2




# Reads in MNIST dataset
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr


# logistic regression hypothesis function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Softmax classifier hypothesis function
def hypo(value):
	# To prevent overflow subract the max number from each element in the array
	constant = value.max()
	return (np.exp(value - constant)/ np.reshape(np.sum(np.exp(value - constant), axis = 1), (len(value), 1)))


def reg_cost(theta_all, inputs, outputs):
	#First we need to decompose our list of theta weights into their individual matrices because these are used for the regularization
	W_1, W_2, B_1, B_2 = seperate(theta_all)

	#Next we need to feed forward to propagate to the output layer so we can compute our cost function J(W,B)
	a_3, a_2 = feed_forward(theta_all, inputs)

	#Computes cost function (with regularization)
	first = np.sum((-1.0 / float(len(inputs))) * np.multiply(outputs, np.log(a_3)))
	second = (l / (2.0)) * (np.sum(np.multiply(W_1, W_1)))
	third = (l / (2.0)) * (np.sum(np.multiply(W_2, W_2)))
	total_cost = first + second + third

	return total_cost


def feed_forward(theta_all, inputs):
	
	#For this exercise because our softmax classifier has a hidden layer we must first use the logistic 	
	#regression hypothesis function (sigmoid) to propagate from the input to the hidden layer.
	#Then from the hidden layer we use the softmax hypothesis function(hypothesis) to go to the output layer. 
	
	#First let's decompose our theta_all
	W_1,W_2,B_1,B_2 = seperate(theta_all)
	
	a_2 = sigmoid(np.dot(inputs, W_1.T) + np.tile(np.ravel(B_1), (len(inputs), 1)))    # (m, 36) matrix

	a_3 = hypo(np.dot(a_2, W_2.T) + np.tile(np.ravel(B_2), (len(inputs), 1)))       # (m, 10) matrix

	return a_3, a_2


def backprop(theta_all, inputs, outputs):
	#This function calculates the individual derivatives for each theta term.
	global global_step
	
	global_step+= 1
	if (global_step% 50 == 0):
		print 'Global step: %g' %(global_step)

	# First we need to take our long theta list and unroll it into the individualized theta matrices.
	W_1, W_2, B_1, B_2 = seperate(theta_all)
	# Feeding forward is second because we need to use our activations for the output and hidden layer to compute our error (delta terms)
	a_3, a_2 = feed_forward(theta_all, inputs)

	#First step towards computing error terms. 
	error = outputs - a_3
	arr_ones = np.ones((len(a_2), 1))
	a_2_1 = np.hstack((arr_ones, a_2))
	theta_2 = np.hstack((B_2, W_2))

	arr_ones = np.ones((len(inputs), 1))
	inputs_1 = np.hstack((arr_ones, inputs))
	theta_1 = np.hstack((B_1, W_1))

	# Calculates the partial derivatives for each theta term. 
	derivative = a_2* (1-a_2)
	delta_2 = np.multiply(np.dot(W_2.T, (error).T).T, derivative)

	Delta_W_1 = np.dot(delta_2.T, inputs_1)         # (36, ? + 1)
	Delta_W_2 = np.dot((error).T, a_2_1)     # (10, 37)

	Grad_W_1 = (-1.0 / len(inputs)) * Delta_W_1 + l * theta_1
	Grad_W_2 = (-1.0 / len(inputs)) * Delta_W_2 + l * theta_2

	# Now we take the gradient calculated for each theta matrix and roll them into individual lists. Then we concatenate them together in the same order that theta_all has. 
	Grad_B_1 = np.ravel(Grad_W_1[:, : 1])
	Grad_B_2 = np.ravel(Grad_W_2[:, : 1])
	Grad_W_1 = np.ravel(Grad_W_1[:, 1: ])
	Grad_W_2 = np.ravel(Grad_W_2[:, 1: ])

	Combined_Grad = np.concatenate((Grad_W_1, Grad_W_2, Grad_B_1, Grad_B_2))
	return Combined_Grad

# Set up our weights and bias terms
def create_weights_bias():
	
	epsilon = 0.12
	
	W_1 = np.random.rand(length_hidden,features_pooled) #36x400 matrix
	W_1 = W_1*2*epsilon - epsilon
	W_2 = np.random.rand(num_classes,length_hidden) #10x36 matrix
	W_2 = W_2*2*epsilon - epsilon

	B_1 = np.random.rand(length_hidden,1) #36x1 matrix
	B_1 = B_1*2*epsilon - epsilon
	B_2 = np.random.rand(num_classes,1) #10x1 matrix
	B_2 = B_2*2*epsilon - epsilon

	theta_all = np.concatenate((np.ravel(W_1),np.ravel(W_2),np.ravel(B_1),np.ravel(B_2)))

	return theta_all




#The first step is to grab our training and testing set of Convolved and Pooled features 


train = np.genfromtxt('output_folder/MNISTTrain_Set_CONVED_AND_POOLED_STEP_25_PoolDim_710_B_0.5_Rho_0.05.out')
train = np.reshape(train, (100, 60000, pool_Dim, pool_Dim))
train = np.swapaxes(train, 0, 1)
train = np.reshape(train, (training_size, length_hidden_old*pool_Dim**2) )  # (60k, 400)
print 'Dimensions of train', train.shape

test = np.genfromtxt('output_folder/MNISTTest_Set_CONVED_AND_POOLED_STEP_25_PoolDim_710_B_0.5_Rho_0.05.out')
test = np.reshape(test, (100, 10000, pool_Dim, pool_Dim))
test = np.swapaxes(test, 0, 1)
test = np.reshape(test, (testing_size,  length_hidden_old*pool_Dim**2))   # (10k, 400)
print 'Dimensions of test', test.shape


#Next we want to load in our MNIST training and testing dataset as well

#Training data set
training_images = read_idx('data/train-images-idx3-ubyte.gz', training_size)
training_images = training_images / 255.0 #This helps normalize our data
training_images = np.reshape(training_images, (training_size, features))
training_labels = read_idx('data/train-labels-idx1-ubyte.gz', training_size)
training_labels = np.reshape(training_labels, (training_size, 1))


#Testing data set
testing_images = read_idx('data/t10k-images-idx3-ubyte.gz', testing_size)
testing_images = testing_images / 255.0 #This helps normalize our data
testing_images = np.reshape(testing_images, (testing_size, features))
testing_labels = read_idx('data/t10k-labels-idx1-ubyte.gz', testing_size)
testing_labels = np.reshape(testing_labels, (testing_size, 1))

#Next let's create our weights and biases
theta_all = create_weights_bias()

#Next let's make our y matrix 
y_vals_mat = y_as_matrix(training_labels,training_size)
print('shape of y matrix is ')
print(y_vals_mat.shape)


print 'Initial Cost: ' 
print(reg_cost(theta_all, train, y_vals_mat))
#We typically get an initial cost value of 2.31 which indicates we have a working cost function


#Next we wanna check the gradient
'''
print scipy.optimize.check_grad(reg_cost, backprop, theta, train, y_vals_train)
'''
#check_grad gives us a very low value which is indicative of a good working backprop function!


# Minimize the cost value
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta_all, method = 'L-BFGS-B', tol = 1e-4, jac = backprop, args = (train, y_vals_mat)) 
best_thetas = minimum.x

print ('Minimized cost function: ')
print(reg_cost(best_thetas, train, y_vals_mat))



#Next let's write out these optimal thetas to a different file
output_name = 'output_folder/optimal_thetas_Soft_Max_l_' + str(l) + '.out'
np.savetxt(output_name, best_thetas,delimiter= ',')


#Finally we want to calculate the Percentages that each number was correctly guessed!


#First we feed forward our theta_values and pooled and convolved test set images to get to output layer
best_guess, a_2_best = feed_forward(best_thetas, test)

# Second we want to find which digit had the highest output (hypothesis) value for each test sample
largest_prob = np.zeros((len(best_guess), 1))
for i in range (len(best_guess)):
	largest_prob[i, 0] = np.argmax(best_guess[i, :])

#Now we can easily count up how many times we correctly guessed the output for each digit
correctly_guessed_digit = np.zeros((num_classes))
for i in range(testing_size):
	if (testing_labels[i] == largest_prob[i]):
		digit = testing_labels[i]
		correctly_guessed_digit[digit] = correctly_guessed_digit[digit] + 1.0


# Next it is important to know how many of each type of digit our test set had to calculate percentages correctly
num_digits = np.zeros((num_classes, 1))
for i in range(num_classes):
	for j in range(testing_size):
		if (testing_labels[j] == i):
			num_digits[i] = num_digits[i] + 1

# Calculate the percentage
print('Calculating Percentages: ')
for i in range(num_classes):
	correctly_guessed_digit[i] = (correctly_guessed_digit[i] / num_digits[i]) * 100
	print(correctly_guessed_digit[i])


























