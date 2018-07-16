#Purpose: The purpose of this code is to correctly implement the softmax classifier and to classify images on the test set
#Created on: 6/14/2018
#Created by: Nick Kyriacou

##################### IMPORTING PACKAGES #####################3
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
######################### GLOBAL VARIABLE DEFINITION #########################

gStep = 0
epsilon = 0.12
features = 784
hidden_layer_size = 200
num_classes = 10
P = 0.1
B = 4
l = 3e-5
m = 60000
#################### DATA PREP FUNCTION ###############################

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
		# Obtain the data values and convert them from arrays to lists
	datx = read_ids('train-images-idx3-ubyte.gz', 60000)
	daty = read_ids('train-labels-idx1-ubyte.gz', 60000)

	# Get the data in matrix form
	datx = np.ravel(datx).reshape((60000, 784))

	# Stick the data and labels together for now
	dat = np.hstack((daty.reshape(60000, 1), datx))
	
	# If the user wants the whole dataset, we can send it back now, after shuffling
	if string == '09':
		np.random.seed(5)	# Some random seed
		np.random.shuffle(dat)
		return dat[:,1:]/255.0, col(dat, 0)


	# Organize the data with respect to the labels
	ind = np.argsort( dat[:,0] ).astype(int)
	ordDat = np.zeros(dat.shape)
	for i in range(len(ind)):
		ordDat[i] = dat[ind[i]]
	#ordDat = dat[dat[:,0].argsort()]

	# Find the index of the last 4. For some reason, this is a 1 element array still, so we choose the only element [0]
	last4Ind = np.argwhere(col(ordDat, 0)==4)[-1][0]
	# Seperate the data
	dat04 = ordDat[0:last4Ind+1]
	dat59 = ordDat[last4Ind+1: ]

	# Reorder the data
	np.random.seed(7)	# Some random seed
	np.random.shuffle(dat04)
	np.random.shuffle(dat59)

	if string == '04':
		return dat04[:,1:]/255.0, col(dat04, 0)
	elif string == '59':
		return dat59[:,1:]/255.0, col(dat59, 0)
	else:
		print 'Error, input is not "04" or "59" or "09"'

def col(matrix, i): #Returns column of a certain index. Borrowed from Suren for simplicity in not having to redo data-prep code
	return np.asarray([row[i] for row in matrix])





####################### FUNCTIONS #####################################



def hypothesis(W, B, x):
	Max = np.amax(np.matmul(W, x.T) + B)
	numer = np.exp( np.matmul(W, x.T) + B - Max )	#This method of finding the maximum value and subtracted all numbers exponeniated by this constant max value is helpful is dealing with overflow errors.
	denom = np.asarray([np.sum(numer, axis=0)])
	return (numer/denom).T


# Calculate the Hypothesis (layer 3) using just layer 1 by feeding forward
def Feed_Forward(theta_1, theta_2, inputs):

	W_1 = np.reshape(theta_1[0:hidden_layer_size*features],(hidden_layer_size,features) )
	B_1 = np.reshape(theta_1[hidden_layer_size*features:],(hidden_layer_size,1) )
	W_2 = np.reshape(theta_2[0: num_classes*hidden_layer_size],(num_classes,hidden_layer_size) ) 
	B_2 = np.reshape(theta_2[num_classes*hidden_layer_size:],(num_classes,1) ) 
	# a_2 has dimensions of (input_size x hidden_layer_size)
	a_2 = hypothesis(W_1, B_1, inputs)
	# a_3 has dimensions of (input_size x num_classes)
	a_3 = hypothesis(W_2, B_2, a_2)
	return a_3, a_2

# Calculate the regularized Cost J(theta)
def RegJCost(theta_2, theta_1, inputs, ymat):
	# Forward Propagate
	a_3, a_2 = Feed_Forward(theta_1, theta_2, inputs)
	# Seperate and reshape the Theta values
	W_2 = np.reshape(theta_2[0: num_classes*hidden_layer_size],(num_classes,hidden_layer_size) ) 
	B_2 = np.reshape(theta_2[num_classes*hidden_layer_size:],(num_classes,1) ) 
	# returns cost function 
	first = np.multiply(np.log(a_3), ymat)
	second = np.sum(first)*(-1.0/float(len(inputs)))
	third = l*0.5*np.sum(W_2**2)
	cost = second+third
	return (cost)


def back_prop(theta_2, theta_1, inputs, ymat):
	# This counts the number of iterations and writes out successive thetas after every 200 iterations in case code is killed
	global gStep
	gStep += 1
	if gStep % 50 == 0:
		print 'Global Step: ', gStep, 'with JCost: ',  RegJCost(theta_2, theta_1, inputs, ymat)
	if gStep % 200 == 0:
		print 'Saving Global Step : ', gStep
		saveW(W_2)

	# Forward Propagate
	a_3, a_2 = Feed_Forward(theta_1, theta_2, inputs)	# DIMENSIONS: a_2 = (input_size x hidden_layer (200) ) & a_3 (input_size x num_classes (10))
	# Seperate and reshape the theta_2 values because these are what we want to optimize for the softmax classifier

	
	W_2 = np.reshape(theta_2[0: num_classes*hidden_layer_size],(num_classes,hidden_layer_size) ) 
	B_2 = np.reshape(theta_2[num_classes*hidden_layer_size:],(num_classes,1) ) 
	
	# WE MUST recombine B_2 and W_2 into a theta_2 matrix again because we use this matrix to calculate gradients
	theta_2_all = np.hstack((B_2, W_2))
	# Attach a column of 1's onto a_2
	ones = np.ones((len(inputs),1))
	a_2_int = np.hstack((ones, a_2))
	# Calculate the derivative for both W2 and b2 at the same time. Breaks up calculation so its eaasier to follow
	output_error = (ymat - a_3)
	second = np.matmul(output_error.T,a_2_int)*(-1.0 / len(inputs))
	third = l*theta_2_all

	Delta_W_2 = second + third	# (g.num_classes, g.hidden_layer_size)
	# Take the gradients and seperate them individually into that for W_2 and B_2, roll them up into a list and feed it back to cost function. 
	theta_list = np.concatenate(( np.ravel(Delta_W_2[:,1:]),np.ravel(Delta_W_2[:,:1]) ))
	return theta_list


# Generate the y-matrix that we will use for calculations in our cost function. 
def GenYMat(yvals):
	yvals = np.ravel(yvals)
	ymat = np.zeros((len(yvals), 10))
	for i in range(len(yvals)):
		for j in range(10):
			if yvals[i] == j:
				ymat[i][j] = 1
	return ymat

############################## MAIN CODE ################################
dat, y = PrepData('04')


dat = dat[:m, :]	# len(y)/2 For 04, 59 testing
y = y[:m]


file_name = 'output_folder/Best_Weights_Rho' + str(P) + 'Lambda' + str(l) + 'Beta' + str(B) + '.out' 
#This is the name of the file where we pull our theta weights from
#Now let's pull out our weights and resort them from one long list into individual arrays
best_thetas = np.genfromtxt(file_name,dtype = 'float')


W_1 = np.reshape(best_thetas[0:features*hidden_layer_size],(hidden_layer_size,features)) #DIMENSIONS: (200,784) (HIDDEN_layer,features)
B_1 = np.reshape(best_thetas[features*hidden_layer_size*2:features*hidden_layer_size*2 + hidden_layer_size],(hidden_layer_size,1)) #DIMENSIONS: (200,1) (HIDDEN_LAYER,1)

W_2 = np.reshape(best_thetas[features*hidden_layer_size:features*hidden_layer_size*2],(features,hidden_layer_size)) #DIMENSIONS: ( 784,200 ) (features,HIDDEN_LAYER)
B_2 = np.reshape(best_thetas[features*hidden_layer_size*2 + hidden_layer_size:features*hidden_layer_size*2 + hidden_layer_size + features],(features,1)) #DIMENSIONS: (784,1) (Features,1)


#Next we need to randomly reinitialize our theta 2 weights
W_2 = np.random.rand(num_classes,hidden_layer_size)
W_2 = W_2*2*epsilon - epsilon
B_2 = np.random.rand(num_classes,1)
B_2 = B_2*2*epsilon - epsilon
#repackaging these weights into the long theta lists we will send through our cost and backprop functions
theta_1 = np.concatenate((np.ravel(W_1),np.ravel(B_1) ))
theta_2 = np.concatenate((np.ravel(W_2),np.ravel(B_2) ))


ymat = GenYMat(y)

print 'Initial W JCost: ', RegJCost(theta_2, theta_1, dat, ymat) 
# Checking the gradient, the gradient check was quite low (circa 1e-4 which indicated a sufficiently working backprop)
'''
print scipy.optimize.check_grad(RegJCost, back_prop, WA2, WA1, dat, ymat)
'''
# Calculate the best theta values for a given j and store them. Usually tol=10e-4. usually 'CG'


res = minimize(fun=RegJCost, x0= theta_2, method='L-BFGS-B', tol=1e-4, jac=back_prop, args=(theta_1, dat, ymat) ) # options = {'disp':True}
optimal_theta_2 = res.x

print 'Final W JCost', RegJCost(optimal_theta_2, theta_1, dat, ymat) 




guesses_for_test_data,a_2_bias_test_data, = Feed_Forward(theta_1,optimal_theta_2,dat)



test_set_size = len(dat) ## CHANGE THIS WHEN YOU WANT TO USE A DIFFERENT TEST SET
test_labels = (y) ## CHANGE THIS WHEN YOU WANT TO USE A DIFFERENT TEST SET


total_digit_count_test_set = np.zeros((10,1))
#Let's look at each element of test_labels and figure out how many total we have of each digit
for i in range(test_set_size):
	digit = int(test_labels[i])
	#print(digit)
	#print(total_digit_count_test_set[digit])
	total_digit_count_test_set[digit] = total_digit_count_test_set[digit] + 1


#Repeat the same process as above and find the index of the highest values and then store them in a test_set_sizex1 array
highest_value_neural_network = np.zeros((test_set_size,1))
for k in range(test_set_size):
	highest_value_neural_network[k,0] = ( np.argmax(guesses_for_test_data[k,:]) )
#Now let's find the percentages we are able to identify a number correctly
correctly_guessing_number_neural = np.zeros((10,1))




for y in range(len(highest_value_neural_network)):
	if (highest_value_neural_network[y] == test_labels[y]):
		correctly_guessing_number_neural[int(test_labels[y])] = correctly_guessing_number_neural[int(test_labels[y])] + 1
#Adding in some scaling factor to turn counts into percentages 
scaling_down_neural = (correctly_guessing_number_neural)
for s in range(10):
	scaling_down_neural[s] = 100.0*(scaling_down_neural[s] / float(total_digit_count_test_set[s]) )
print('Lambda = ',l)
print('Beta = ', B)
print('Rho = ', P )
print("Percentages each number was guessed correctly using a neural network")
print(scaling_down_neural.shape)
print(scaling_down_neural)

