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

##################### FUNCTION DEFINITIONS GO HERE #####################################

def read_ids(filename, n = None): #This function used to read MNIST dataset
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB',f.read(4))
		shape  = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(),dtype = np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
			return arr


def PrepData(string): #This prepares the data to be either ranging from digits 5-9, digits 0-4 or 0-9 (the entire dataset) (Borrowed from Suren)
	# Obtain the data values and convert them from arrays to lists
	training_data = read_ids('train-images-idx3-ubyte.gz', training_sets)
	training_labels = read_ids('train-labels-idx1-ubyte.gz', training_sets)

	# Get the data in matrix form
	training_data = np.ravel(training_data).reshape((training_sets, features))

	# Stick the data and labels together for now
	data = np.hstack((training_labels.reshape(training_sets, 1), training_data))
	
	# If the user wants the whole dataset, we can send it back now, after shuffling
	if string == '09':
		np.random.seed(5)	# Some random seed
		np.random.shuffle(data)
		return data[:,1:]/255.0, col(data, 0)


	# Organize the data with respect to the labels
	index = np.argsort( data[:,0] ).astype(int)
	ordData = np.zeros(data.shape)
	for i in range(len(index)):
		ordData[i] = data[index[i]]
	#ordDat = dat[dat[:,0].argsort()]

	# Find the index of the last 4. For some reason, this is a 1 element array still, so we choose the only element [0]
	last4Ind = np.argwhere(col(ordData, 0)==4)[-1][0]
	# Seperate the data
	data_0_4 = ordData[0:last4Ind+1]
	data_5_9 = ordData[last4Ind+1: ]

	# Reorder the data
	np.random.seed(7)	# Some random seed
	np.random.shuffle(data_0_4)
	np.random.shuffle(data_5_9)

	if string == '04':
		return data_0_4[:,1:]/255.0, col(data_0_4, 0)
	elif string == '59':
		return data_5_9[:,1:]/255.0, col(data_5_9, 0)
	else:
		print 'Error, input is not "04" or "59" or "09"'


def col(matrix, i): #Returns column of a certain index. Borrowed from Suren for simplicity in not having to redo data-prep code
	return np.asarray([row[i] for row in matrix])

########## Machine Learning Algorithm Implementation ##################

def create_weights_and_bias(): #This function creates all W's and B's and combines them into an unrolled list and returns said list
	#It is important that all weights are randomly created on an interval [-epsilon,epsilon] very close to zero
	theta1_random = np.random.rand(length_hidden,features) #200x784 matrix
	print(theta1_random.shape)
	theta1_random = theta1_random*2*epsilon - epsilon
	theta2_random = np.random.rand(features,length_hidden) #784x200 matrix
	print(theta2_random.shape)
	theta2_random = theta2_random*2*epsilon - epsilon
	
	bias_1 = np.random.rand(length_hidden,1) #200x1 matrix
	print(bias_1.shape)
	bias_1 = bias_1*2*epsilon - epsilon
	bias_2 = np.random.rand(features,1) #784x1 matrix
	print(bias_2.shape)
	bias_2 = bias_2*2*epsilon - epsilon
	theta_all = np.concatenate((np.ravel(theta1_random), np.ravel(theta2_random), np.ravel(bias_1),np.ravel(bias_2)))
	#print('in creating weights')
	#print(theta_all.shape)
	return(theta_all)
	
def seperate(theta_all): #This function will take a combined theta vector and seperate it into 4 of its specific components
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features))
	B_1 = np.reshape(theta_all[2*features*length_hidden:2*features*length_hidden + length_hidden],(length_hidden,1))
	W_2 = np.reshape(theta_all[features*length_hidden:2*features*length_hidden],(features,length_hidden))
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1))
	#Now that we have seperated each vector and reshaped it into usable format lets return each weight
	#print('in seperate')
	#print(W_1.shape)
	#print(W_2.shape)
	#print(B_1.shape)
	#print(B_2.shape)
	return(W_1,B_1,W_2,B_2)

def feed_forward(theta_all,xvals):
	#First we must unravel our theta1 and theta2 and reshape them into the correct dimensions
	W_1,B_1, W_2, B_2 = seperate(theta_all)
	#print('inside feed forward')
	#print(W_1.shape)
	#print(B_1.shape)
	#print(xvals.shape)
	a_2 = sigmoid(W_1,B_1,xvals) #Calculates a_2 as a matrix
	a_2 = a_2.T
	#print(a_2.shape)
	#print('that was a_2')
	a_3 = sigmoid(W_2,B_2,a_2)
	a_3 = a_3.T
	#print(a_3.shape)
	#print('did this:feed_forward')
	return(a_3,a_2)


def sigmoid(W,B,inputs): #Add surens fix if there are overflow errors
	z = np.matmul(W, inputs.T) + B #Should be a 25x10000 matrix
	#print('in sigmoid')
	#print(z.shape)
	hypo = 1.0/(1.0 + np.exp(-1.0*z))
	return(hypo)

# Calculate the regularized Cost J(theta)
def sparse_cost_function(theta_all, inputs):
	# Forward Propagates to output layer
	a_3, a_2 = feed_forward(theta_all, inputs)
	# Seperate and reshape the Theta values
	W_1, B_1, W_2, B_2 = seperate(theta_all)
	# Calculate Sparsity contribution. Hehe, phat sounds like fat (stands for p hat)
	p_hat = (1.0 / float(len(inputs)))*np.sum(a_2, axis=0) # 200 len vector
	diff = a_3 - inputs
	# Calculate Cost as a function of W,B, lambda, and Beta
	Cost_first = (0.5/float(len(inputs)))*np.sum((diff)**2)
	Cost_second = Cost_first + (0.5/float(len(inputs)))*l * (np.sum(W_1**2)+np.sum(W_2**2))
	#print('phat is', p_hat)
	#print(np.log(P / p_hat) ) This part ends up being ok
	#print(np.log((1-P)/(1-p_hat)) )
	Cost_third = Cost_second + Beta * np.sum(   P*np.log(P / p_hat) + (1-P)*np.log((1-P)/(1-p_hat))  )
	return Cost_third

# Calculate the gradient of cost function for all values of W1, W2, b1, and b2
def back_prop(theta_all, inputs):
	
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
	# Creating (Capital) Delta matrices
	DeltaW1 = np.zeros(W_1.shape)			# (g.f2, g.f1)
	DeltaW2 = np.zeros(W_2.shape)			# (g.f1, g.f2)
	Deltab1 = np.zeros(B_1.shape)			# (g.f2, 1)
	Deltab2 = np.zeros(B_2.shape)			# (g.f1, 1)
	# Calculate error term for each element in the dataset and add it's contributions to the capital Delta terms

	
	# Calculate Sparsity contribution to delta2
	p_hat = (1.0 / float(len(inputs)))*np.sum(a_2, axis=0)
	K_L = Beta * ( -P/p_hat + (1-P)/(1-p_hat)	)
	delta3 = np.multiply( -1*(inputs- a_3), a_3*(1-a_3) )
	delta2 = np.multiply( np.matmul(delta3, W_2) + K_L.reshape(1, length_hidden), a_2*(1-a_2) )

	#Initializes the Gradient for each vector
	Grad_W_1 = np.dot(delta2.T, inputs) 	# (25, 64)
	Grad_W_2 = np.dot(delta3.T, a_2)     	# (64, 25)
	Grad_B_1 = np.mean(delta2, axis = 0) # (25,) vector
	Grad_B_2 = np.mean(delta3, axis = 0) # (64,) vector
	#Now let's calculate the grad for our W_1 and W_2
	Grad_W_1 = (1.0/float(len(inputs)))*(Grad_W_1+ l*W_1)
	Grad_W_2 = (1.0/float(len(inputs)))*(Grad_W_2 + l*W_2)
	Combined_Grad = np.concatenate((np.ravel(Grad_W_1),np.ravel(Grad_W_2),np.ravel(Grad_B_1),np.ravel(Grad_B_2)))
	return ( Combined_Grad )

############################## MAIN CODE STARTS HERE ###############################################

#First let us set up values for our global variables

training_sets = 60000 #60000 training samples
features = 784 #784 features per sample (excluding x-int bias we add) because we have set of 28x28 pixellated images
l = .005 #Lambda constant
num_classes = 10 #There are 10 different possible classifications, This should and will change based on 
length_hidden = 200
Beta = 3
P = 0.1
test_set_size = 10000 #test set has 10000 entries
epsilon = .1
output_name = 'output_folder/optimalweights_lambda_' + str(l) + '_Beta_' + str(Beta) + '_Rho_' + str(P) + '.out'

#Next Let's call the function prep_data to get three data sets

#First let's get our randomized data-set of 5-9 inputs
training_data_5_9, training_labels_5_9 = PrepData('59')
training_labels_5_9 = np.resize(training_labels_5_9,(len(training_labels_5_9),1))
print(training_data_5_9.shape)
print(training_labels_5_9.shape)
#Second let's grab our randomized data-set of 0-4
training_data_0_4,training_labels_0_4 = PrepData('04')
training_labels_0_4 = np.resize(training_labels_0_4,(len(training_labels_0_4),1))
print(training_data_0_4.shape)
print(training_labels_0_4.shape)
#Finally we can also grab the entire data-set of 0-9 in case there is any use for that
training_data_0_9, training_labels_0_9 = PrepData('09')
training_labels_0_9 = np.resize(training_labels_0_9,(len(training_labels_0_9),1))
print(training_data_0_9.shape)
print(training_labels_0_9.shape)

#Now that we have all our data correctly structured let's make our weights of thetas
theta_all = create_weights_and_bias()
print('theta_all')
print(theta_all.shape)
###Something to think about later on. Whether or not I have to normalize my data...
print(784*200)
print(theta_all[0:784*200].shape)
#First let's see what the initial cost is
print('initial cost is')
print(sparse_cost_function(theta_all,training_data_5_9))
#Initial cost function turns out to be quite high (~200), now let's check the gradient
#print(scipy.optimize.check_grad(sparse_cost_function,back_prop,theta_all,training_data_5_9))
#Grad gives a small value ~ <1e^-4 so we can ignore this for now


#Now let's move onto optimizing our weights
print('Finding weights that optimize the sparse cost function')
optimize_time = scipy.optimize.minimize(fun = sparse_cost_function,x0 = theta_all,method = 'CG',tol = 1e-4,jac = back_prop,args = (training_data_5_9,))
print('optimal cost function is')
optimal_thetas = optimize_time.x
optimal_thetas_cost = sparse_cost_function(optimal_thetas,training_data_5_9)
print(optimal_thetas_cost)

#Next let us write out these optimal thetas to an output file
np.savetxt(output_name,optimal_thetas,delimiter = ',')




################ VISUALIZATION OF ACTIVATION UNITS OF HIDDEN LAYERS#######################

#we want to take these optimal thetas and feed foward twice (primarily to get the activation units in the hidden layer)
a_3_best,a_2_best = feed_forward(optimal_thetas,training_data_5_9)


print(a_3_best.shape)
print(a_2_best.shape)
