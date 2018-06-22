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
	p_hat = (1.0 / 10000.0)*np.sum(a_2, axis=0) # 25 len vector
	diff = a_3 - inputs
	# Calculate Cost as a function of W,B, lambda, and Beta
	Cost_first = (0.5/10000.0)*np.sum((diff)**2)
	Cost_second = Cost_first + (0.5/10000.0)*l * (np.sum(W_1**2)+np.sum(W_2**2))
	Cost_third = Cost_second + Beta * np.sum(   P*np.log(P / p_hat) + (1-P)*np.log((1-P)/(1-p_hat))  )
	return Cost_third

# Calculate the gradient of cost function for all values of W1, W2, b1, and b2
def back_prop(theta_all, inputs):
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
	p_hat = (1.0 / 10000.0)*np.sum(a_2, axis=0)
	K_L = Beta * ( -P/p_hat + (1-P)/(1-p_hat)	)
	delta3 = np.multiply( -1*(inputs- a_3), a_3*(1-a_3) )
	delta2 = np.multiply( np.matmul(delta3, W_2) + K_L.reshape(1, length_hidden), a_2*(1-a_2) )

	#Initializes the Gradient for each vector
	Grad_W_1 = np.dot(delta2.T, inputs) 	# (25, 64)
	Grad_W_2 = np.dot(delta3.T, a_2)     	# (64, 25)
	Grad_B_1 = np.mean(delta2, axis = 0) # (25,) vector
	Grad_B_2 = np.mean(delta3, axis = 0) # (64,) vector
	#Now let's calculate the grad for our W_1 and W_2
	Grad_W_1 = (1.0/10000.0)*(Grad_W_1+ l*W_1)
	Grad_W_2 = (1.0/10000.0)*(Grad_W_2 + l*W_2)
	Combined_Grad = np.concatenate((np.ravel(Grad_W_1),np.ravel(Grad_W_2),np.ravel(Grad_B_1),np.ravel(Grad_B_2)))
	return ( Combined_Grad )



#The data starts out on the range of values from -1 to 1 and thus it is important to normalize them such that they all fall between 0 and 1 because these are the valles our sigmoid takes in. 
def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	normed = ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin
	return(normed)

#MAIN SECTION OF CODE STARTS HERE


l = 1 #Lambda constant 
length_hidden = 25 #Length of hidden layer (without bias added)
epsilon = .12 #Used in gradient check function
training_sets = 100 #10000 samples
features = 64 #64 features comes from 8x8 hidden layer
Beta = float(1) #Weight of Sparsity Parameter
P = float(0.01) #Sparsity Parameter


#For a sparse auto encoder our x = y
data = np.genfromtxt('output_folder/10000Random8x8.out',dtype = float)

# Roll up data into matrix. The Normalization restricts each value to be [0,1], because we are using the sigmoid hypothesis function
data = np.asarray(data.reshape(10000,64))
data = data[0:10000, :]
# Normalize each image
for i in range(10000):
	data[i] = Norm(data[i])

# Prepare the W matrices and B matrices by randomly assinging them values between [-epsilon,epsilon] and linearize them
theta_all = create_weights_and_bias()
W_1,B_1, W_2, B_2 = seperate(theta_all)

temp = data[0:100,]
# CALCULATING IDEAL W MATRICES
# Check the cost of the initial W matrices
print 'Initial W JCost: ', sparse_cost_function(theta_all, data) 

# Check the gradient. Go up and uncomment the import check_grad to use. ~1.0574852249e-05 for randomized Ws and bs
#print('gradient is')
#print scipy.optimize.check_grad(sparse_cost_function, back_prop, theta_all, data)

#Next let us optimize the cost function!!
print('Lambda is ', l)
print('now we are optimizing the cost function')
optimize_time = scipy.optimize.minimize(fun = sparse_cost_function,x0 = theta_all,method = 'CG',tol = 1e-4,jac = back_prop,args = (temp,))
print('optimal cost function is')
optimal_thetas = optimize_time.x
optimal_thetas_cost = sparse_cost_function(optimal_thetas,temp)
print(optimal_thetas_cost)


# Save our theta weights array to an output file to be used later
optimal_thetas  = np.ravel(optimal_thetas)

name = 'output_folder/optimalweights_l' + str(l) +'_Beta' + str(Beta) + '_Rho'+ str(P) +'.out'
np.savetxt(name, optimal_thetas, delimiter = ',')



#First let us calculate the optimal a_2 and a_3 values
#Ultimately we want to display the input and hidden layer that gives us the maximum output layer
a_3_best, a_2_best = feed_forward(optimal_thetas,data)



# SHOW IMAGES
hspaceAll = np.asarray([ [0 for i in range(53)] for j in range(5)])
picAll = hspaceAll




# We also want a picture of the input that maximally activates  each node in the hidden layer. Borrowed from Suren for simplicity
W_1, B_1, W_2, B_2 = seperate(optimal_thetas)
W1Len = np.sum(W_1**2)**(-0.5)
X = W_1 / W1Len			
X = Norm(X)

picX = np.zeros((25,8,8))
for i in range(25):
	picX[i] = np.reshape(np.ravel(X[i]), (8,8))

hblack = np.asarray([ [1 for i in range(52)] for j in range(2)])
vblack = np.asarray([ [1 for i in range(2)] for j in range(8)])

picAll = hblack
for i in range(5):
	pici = np.concatenate((vblack, picX[5*i+0], vblack, picX[5*i+1], vblack, picX[5*i+2], vblack, picX[5*i+3], vblack, picX[5*i+4], vblack), axis = 1)
	print('shapes')
	print(picAll.shape)
	print(pici.shape)
	print(hblack.shape)
	picAll = np.vstack((picAll, pici, hblack))

# Display the pictures
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.savefig('Picture_Folder/'+'activations_for_node_hidden_layer'+'Lamb_'+str(l)+'_Beta_'+str(Beta) +'_Rho_'+str(P) +'.png',transparent=False, format='png')
plt.show()




