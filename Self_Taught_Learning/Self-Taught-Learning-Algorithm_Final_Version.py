#Purpose:This is my SELF TAUGHT LEARNING ALGORITHM. The purpose of this code is to teach itself to classify handwritten digits using the MNIST dataset.First it will optimize its weights (W,B) on an unlabeled data-set. 
#Created by: Nick Kyriacou
#Created on: 6/13/2018


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
global output_name #This is the name of the file we will write out to
global_iterations = 0 #Number of iterations



global_step = 0
features  = 784   #(28x28 pixel images)
hidden_layer_size = 200 #Number of nodes in hidden layer
P = 0.1 #FIXME
l = 3e-7 #FIXME   # weight decay parameter (3e-2)
B = 0.3 #FIXME     # weight of sparsity penalty term (3)



# Set up the filename we want to use
filename = 'output_folder/Best_Weights_Rho' + str(P) + 'Lambda' + str(l) + 'Beta' + str(B) + '.out'

######################## Function Definitions ################################

def col(array, i):
	return np.asarray([row[i] for row in array])

def get_data(size, string):
	# Extract the MNIST training data sets. Borrowed from Matt's Code
	x_vals = read_idx('train-images-idx3-ubyte.gz', size)
	x_vals = x_vals / 255.0
	x_vals = np.reshape(x_vals, (x_vals.shape[0], (x_vals.shape[1] * x_vals.shape[2])))
	y_vals = read_idx('train-labels-idx1-ubyte.gz', size)
	y_vals = np.reshape(y_vals, (len(y_vals), 1))
	print x_vals.shape
	print y_vals.shape
	
	data = np.hstack((y_vals,x_vals))
	
	# We need to organize the data with respect to the labels
	index = np.argsort(data[:,0]).astype(int)
	data_order = np.zeros(data.shape)
	for i in range(len(index)):
		data_order[i] = data[index[i]]
	
	# Find where we need to split our array
	last_indices = np.argwhere(col(data_order, 0) == 4)[-1][0]

	# Separate our data from 0-4 and 5-9
	num04 = data_order[0:last_indices + 1]
	num59 = data_order[last_indices + 1: ]

	# Now we need to shuffle the data
	np.random.seed(7)
	np.random.shuffle(num04)
	np.random.shuffle(num59)

	if (string == '59'):
		print "59"
		print num59[:,1:].shape
		print np.reshape(col(num59, 0), (len(col(num59, 0)), 1)).shape
		return num59[:,1:], np.reshape(col(num59, 0), (len(col(num59, 0)), 1))
	
	elif (string == '04'):
		print "04"
		print num04[:,1:].shape
		print np.reshape(col(num04, 0), (len(col(num04, 0)), 1)).shape
		return num04[:,1:], np.reshape(col(num04, 0), (len(col(num04, 0)), 1))





# Reading in MNIST data files	
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr

# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Regularized cost function
def reg_cost(theta_all, inputs, y):
	# Change our weights and bias values back into their original shape
	W_1, W_2, B_1, B_2 = reshape(theta_all)

	# feeding forward
	a_3, a_2 = feed_forward(W_1, W_2, B_1, B_2, inputs)

	# Find the average activation of each hidden unit averaged over the training set
	rho_hat = (1.0 / len(inputs)) * np.sum(a_2, axis = 0)         # (hidden_layer_size,1) matrix

	# Calculate the cost
	KL_divergence = B * np.sum((P * np.log(P / rho_hat) + (1 - P) * np.log((1 - P) / (1 - rho_hat))))

	# Calculate the cost
	cost_first = (1.0 / (2 * input_size)) * np.sum(np.multiply((a_3 - y), (a_3 - y)))
	cost_second = (l / (2.0)) * (np.sum(np.multiply(W_1, W_1)))
	cost_third = (l / (2.0)) * (np.sum(np.multiply(W_2, W_2)))
	cost = cost_first + cost_second + cost_third + KL_divergence

	return cost

# Feedforward
def feed_forward(W_1, W_2, B_1, B_2, inputs):

	'''
	We will be running our sigmoid function twice.
	Tile function allows us to duplicate our rows to the proper dimensions without requiring a for loop.
	This enables each row in our dot product to receive the same bias term. If it were a (25, 10000) array it is equivalent to adding 
	our bias column to each dot product column with just + b1 (since b1 starts as a column).
	'''
	z = np.dot(inputs, W_1.T) + np.tile(np.ravel(B_1), (input_size, 1))
	a_2 = sigmoid(z)    # (input_size, hidden_layer_size) matrix

	# To get to output layers
	z = np.dot(a_2, W_2.T) + np.tile(np.ravel(B_2), (input_size, 1))
	a_3 = sigmoid(z)       # (input_size, features) matrix

	return a_3, a_2

# Backpropagation
def back_prop(theta_all, inputs, y):
	# To keep track of our iterations
	global global_step
	global_step += 1
	if (global_step % 50 == 0):
		np.savetxt(filename, theta_all, delimiter = ',')
		print 'Global step: %g' %(global_step)

	# Change our weights and bias values back into their original shape
	W_1, W_2, B_1, B_2 = reshape(theta_all)
	
	a_3, a_2 = feed_forward(W_1, W_2, B_1, B_2, inputs)

	# Find the average activation of each hidden unit averaged over the training set
	p_hat = np.mean(a_2, axis = 0)         # (hidden_layer_size,1) matrix
	p_hat = np.tile(p_hat, (len(inputs), 1))      # (input_size, hidden_layer_size) matrix 

	delta_3 = np.multiply(-(y - a_3), a_3 * (1 - a_3))   # (input_size, features)
	delta_2 = np.multiply(np.dot(delta_3, W_2) + B * (-(P / p_hat) + ((1 - P) / (1 - p_hat))), a_2 * (1 - a_2))
	
	
	# Compute the partial derivatives
	pd_W1 = np.dot(delta_2.T, inputs)  # (hidden_layer_size, features) matrix
	pd_W2 = np.dot(delta_3.T, a_2)     # (features, hidden_layer_size) matrix
	pd_b1 = np.mean(delta_2, axis = 0) # (hidden_layer_size,) matrix
	pd_b2 = np.mean(delta_3, axis = 0) # (features,1) matrix

	del_W1 = (1.0 / float(len(inputs))) * pd_W1 + l * W_1
	del_W2 = (1.0 / float(len(inputs))) * pd_W2 + l * W_2
	del_b1 = pd_b1
	del_b2 = pd_b2

	# Changed the gradients into a one dimensional vector
	del_W1 = np.ravel(del_W1)
	del_W2 = np.ravel(del_W2)
	D_vals = np.concatenate((del_W1, del_W2, del_b1, del_b2))
	return D_vals

# Set up our weights and bias terms
def create_weights_and_bias():
	# Initialize parameters randomly based on layer sizes from interval [-epsilon, epsilon]
	
	epsilon  = 0.12
	W1 = np.random.rand(hidden_layer_size, features)     # (hidden_layer_size, features) matrix
	W1 = W1 * 2 * epsilon - epsilon
	W2 = np.random.rand(features, hidden_layer_size)     # (features, hidden_layer_size) matrix      
	W2 = W2 * 2 * epsilon - epsilon

	# Set up our bias term
	bias1 = np.random.rand(hidden_layer_size, 1)     # (hidden_layer_size, 1) matrix
	bias1 = bias1 * 2 * epsilon - epsilon
	bias2 = np.random.rand(features, 1)    # (features, 1) matrix
	bias2 = bias2 * 2 * epsilon - epsilon


	# theta_all is a long list of all the combined parameters
	theta_all = np.concatenate((np.ravel(W1),np.ravel(W2), np.ravel(bias1), np.ravel(bias2)))	
	
	return theta_all

# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:hidden_layer_size * features], (hidden_layer_size, features))
	W2 = np.reshape(theta[hidden_layer_size * features: 2 * hidden_layer_size * features], (features, hidden_layer_size))
	B1 =np.reshape(theta[2 * hidden_layer_size * features: 2 * hidden_layer_size * features + hidden_layer_size], (hidden_layer_size, 1))
	B2 =np.reshape(theta[2 * hidden_layer_size * features + hidden_layer_size: len(theta)], (features, 1))
	
	return W1, W2, B1, B2

########################### MAIN CODE STARTS HERE  ######################


# HOW MUCH INPUT DATA WE WANT TO GRAB
input_size = 60000

# Extract the full MNIST training data sets
train_0_9 = read_idx('train-images-idx3-ubyte.gz', input_size)
train_0_9 = train_0_9 / 255.0 #The following normalizes the data so that each pixel runs between 0 and 1
train_0_9 = np.reshape(train_0_9, (train_0_9.shape[0], (train_0_9.shape[1] * train_0_9.shape[2])))
labels_train_0_9 = read_idx('train-labels-idx1-ubyte.gz', input_size)
labels_train_0_9 = np.reshape(labels_train_0_9, (len(labels_train_0_9), 1))


# Next let's grab our digits 5 through 9 that we will use to learn features on
train_5_9, labels_train_5_9 = get_data(input_size,'59')

# Need to know how many inputs we have
input_size = len(train_5_9)

# Create our weights and bias terms
theta1 = create_weights_and_bias()

# We want out x = y for our sparse autoencoder
y = train_5_9


#Take a small sample size to not overburden learning algorithm
'''
sample = train[0:10]
y = sample
input_size = len(sample)
print(sample.shape)
print(y.shape)
# Check that our cost function is working
cost_test = reg_cost(theta1, sample, y)
print cost_test
# We had a cost value of 38 (from 20 nodes instead of 200)
# Gradient checking from scipy to see if our back_prop function is working properly. Theta_vals needs to be a 1-D vector.
print scipy.optimize.check_grad(reg_cost, back_prop, theta1, sample, y)
# Recieved a value of 1.2e-4
'''
print 'Cost before minimization: %g' %(reg_cost(theta1, train_5_9, y))

# Minimize the cost value
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta1, method = 'L-BFGS-B', tol = 1e-4, jac = back_prop, args = (train_5_9, y)) #options = {"disp":True}
print minimum
theta_new = minimum.x

print 'Cost after minimization: %g' %(reg_cost(theta_new, train_5_9, y))

# Save to a file to use later
np.savetxt(filename, theta_new, delimiter = ',')

