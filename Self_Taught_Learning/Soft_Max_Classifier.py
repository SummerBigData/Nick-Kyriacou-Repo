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


################################# FUNCTION DEFINITIONS #############################
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

def seperate_updated(theta_all): #This updated seperate function will correctly pull apart W_1,B_1,W_2, AND B_2 now that they are in different orders

	theta_1 = np.reshape(theta_all[0:length_hidden*features + length_hidden],(length_hidden*features + length_hidden,1))
	W_1 = np.reshape(theta_1[0:length_hidden*features],(length_hidden,features))
	B_1 = np.reshape(theta_1[length_hidden*features:len(theta_1)],(length_hidden,1))

	theta_2 = np.reshape(theta_all[length_hidden*features + length_hidden:len(theta_all)],(length_hidden*num_classes + num_classes,1))
	W_2 = np.reshape(theta_2[0:length_hidden*num_classes],(num_classes,length_hidden))
	B_2 = np.reshape(theta_2[length_hidden*num_classes:len(theta_2)],(num_classes,1))
	return(W_1,B_1,W_2,B_2)

def seperate_original(theta_all): #This function will take a combined theta vector and seperate it into 4 of its specific components
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features))
	B_1 = np.reshape(theta_all[2*features*length_hidden:2*features*length_hidden + length_hidden],(length_hidden,1))
	W_2 = np.reshape(theta_all[features*length_hidden:2*features*length_hidden],(features,length_hidden))
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1))
	return(W_1,B_1,W_2,B_2)

def feed_forward(theta_all,xvals): #This is fine and stays the same
	
	#Let's create all our individual weight terms from theta_all
	W_1,B_1,W_2,B_2 = seperate_updated(theta_all)
	a_2 = soft_max_hypo(W_1,B_1,xvals)    # (m, 200) matrix
	
	# Second run
	a_3 = soft_max_hypo(W_2,B_2,a_2)  # (m, 10) matrix
	return(a_3,a_2)


def soft_max_hypo(W,B,inputs): #Add surens fix if there are overflow errors

	Max = np.amax(np.matmul(W, inputs.T) + B)
	numerator = np.exp( np.matmul(W, inputs.T) + B - Max )	# 200 x 15298 for W1, b1
	denominator = np.asarray([np.sum(numerator, axis=0)])


	return(numerator/denominator).T

# Calculate the regularized Cost J(theta)
def soft_max_regularized_cost_function(theta_2,theta_1, inputs,outputs):
	#we need to be able to feed_forward to get our hypothesis values
	#To do this let's recombine our theta_2 and theta_1 values
	theta_all = np.concatenate((np.ravel(theta_1),np.ravel(theta_2)))
	# Forward Propagates to output layer
	a_3, a_2 = feed_forward(theta_all, inputs)
	# Seperate and reshape the Theta values
	W_1, B_1, W_2, B_2 = seperate_updated(theta_all)
	# Calculate Sparsity contribution. Hehe, phat sounds like fat (stands for p hat)
	Cost_first = (-1.0 / float(len(inputs)))*np.sum( np.multiply(np.log(a_3), outputs))
	Cost_second = (l / (2.0)) * (np.sum(W_2**2))
	Cost_total = Cost_first + Cost_second
	return Cost_total

# Calculate the gradient of cost function for all values of W1, W2, b1, and b2
def back_prop(theta_2,theta_1, inputs,outputs):
	
	#We need to keep track of total iterations and every 20 times write out our thetas to the output file
	global global_iterations
	global_iterations = global_iterations + 1 
	if (global_iterations%20 == 0):
		#np.savetxt(output_name,theta_2,delimiter = ',') ######Fix me!
		print('Iteration Number ',global_iterations)

	# Seperate and reshape the W and b values
	theta_all = np.concatenate((np.ravel(theta_1),np.ravel(theta_2)))
	W_1, B_1, W_2, B_2 = seperate_updated(theta_all)
	# Forward Propagate and create an array of ones to attach onto a_2

	a_3, a_2 = feed_forward(theta_all, inputs)	# a2 (g.m x 25), a3 (g.m x 64)
	ones = np.ones((len(a_2),1))
	print(a_3.shape)
	print(a_2.shape)


	a_2_int = np.hstack((ones,a_2))

	# We also want to stack together our W_2 and B_2 

	W_B_2_combined = np.hstack((B_2,W_2))

 	#Now let us begin to calculate the partial derivatives of W_2 (It's easier to break this calculation up into several steps

	first = np.matmul( (outputs - a_3).T, a_2_int )
	second = l * W_B_2_combined 
	gradient_W_2 = ( -1.0/float(len(inputs)) ) * first + second
	
	# Next let's make this gradient a list (1-D vector) so that we can them up and pass back from the back_prop function

	Delta_B_2 = np.ravel(gradient_W_2[:,:1])
	Delta_W_2 = np.ravel(gradient_W_2[:,1:])

	Grad_theta_2 = np.concatenate((Delta_W_2,Delta_B_2))
	
	return(Grad_theta_2)



def y_as_matrix(y,training_sets): #This takes a training_setsx1 vector and makes it into a training_setsx10 matrix.
	y = np.ravel(y)
	y_array = np.zeros((training_sets,10))
	for i in range(len(y)):
		for j in range(10):
			if (y[i] == j):
				y_array[i][j] = 1
			if (y[i] == 10 and j == 0):
				y_array[i][j] = 1
	return(y_array)

################### MAIN CODE STARTS HERE ###################333

#THE FOLLOWING IS A LIST OF THE DIMENSIONS OF EACH WEIGHT MATRIX
#W_1 = 200x784 (length_hiddenxfeatures)
#B_1 = 200x1 (length_hiddenx1)
#W_2 = 10x200 (num_classesxlength_hidden)
#B_2 = 10x1 (num_classesx1)

#First let's load our optimal weights
training_sets = 60000
features  = 784
length_hidden = 200
l = 10
Beta = 1.0
P = 0.05
num_classes = 10
epsilon = 0.12
global_iterations = 0
file_name = 'output_folder/optimalweights_lambda_' + str(l) + '_Beta_' + str(Beta) + '_Rho_' + str(P) + '.out' #This is the name of the file where we pull our theta weights from
optimal_thetas = np.genfromtxt(file_name,dtype = 'float')
print('optimal_thetas')
print(optimal_thetas.shape)

#Next let's read in and process our data in the correct format
training_data_5_9, training_labels_5_9 = PrepData('59')
training_labels_5_9 = np.resize(training_labels_5_9,(len(training_labels_5_9),1))
print(training_data_5_9.shape)
print(training_labels_5_9.shape)
#Then let's grab our randomized data-set of 0-4
training_data_0_4,training_labels_0_4 = PrepData('04')
training_labels_0_4 = np.resize(training_labels_0_4,(len(training_labels_0_4),1))
print(training_data_0_4.shape)
print(training_labels_0_4.shape)
#Finally we can also grab the entire data-set of 0-9 in case there is any use for that
training_data_0_9, training_labels_0_9 = PrepData('09')
training_labels_0_9 = np.resize(training_labels_0_9,(len(training_labels_0_9),1))
print(training_data_0_9.shape)
print(training_labels_0_9.shape)


#Then let's take our W_1 and B_1 weights to get our activation unit layers we solved for earlier

W_1,B_1,W_2,B_2 = seperate_original(optimal_thetas) 
print('shapes')
print(W_1.shape)
print(B_1.shape)
theta_1 = np.concatenate((np.ravel(W_1),np.ravel(B_1)))
print(theta_1.shape)
#Afterwards let's randomly initialize some W_2 and B_2 weights and concatenate them into a theta_2 array
W_2 = np.random.rand(num_classes,length_hidden) #10x200 matrix
W_2 = W_2*2*epsilon - epsilon
print(W_2.shape)
B_2 = np.random.rand(num_classes,1) #10x1 matrix
print(B_2.shape)
B_2 = B_2*2*epsilon - epsilon
theta_2 = np.concatenate((np.ravel(W_2),np.ravel(B_2)))
print(theta_2.shape)
theta_all = np.concatenate((theta_1,theta_2))
print('seperating')
seperate_updated(theta_all)

y_mat =  y_as_matrix(training_labels_0_4,len(training_labels_0_4))


#First to see if our cost function is working we must check our initial cost function value
print ('Initial cost function value is ')
print(soft_max_regularized_cost_function(theta_2,theta_1, training_data_0_4,y_mat) )
#print('checking_grad')
#print(scipy.optimize.check_grad(soft_max_regularized_cost_function,back_prop,theta_2,theta_1,subset,subset_test))

#Next we should optimize our theta2 weights for this function
print('Finding theta_2 weights that optimize the soft max cost function')
print('lambda = ',l)
print('beta= ', Beta)
print('rHO = ', P )
optimize_time = scipy.optimize.minimize(fun = soft_max_regularized_cost_function,x0 = theta_2,method = 'L-BFGS-B',tol = 1e-4,jac = back_prop,args = (theta_1, training_data_0_4,y_mat)     )
print('optimal cost function is')
best_theta_2 = optimize_time.x
optimal_thetas_cost = soft_max_regularized_cost_function(best_theta_2,theta_1,training_data_0_4,y_mat)
print(optimal_thetas_cost)

############ THIS NEXT SECTION ATEMPTS TO CALCULATE PERCENTAGES THAT EACH DATA-SET IS CORRECTLY GUESSED ##################


#Now that we have optimized our weights the next step is to take this set of optimized theta_1 and theta_2 and run them together through our hypothesis function. 
theta_best_all = np.concatenate((np.ravel(theta_1),np.ravel(best_theta_2)))
guesses_for_test_data, a_2_bias_test_data = feed_forward(theta_best_all,training_data_0_4)

#After we have our best guess for each training example let us compare how well we did to the test set
#Next we can find the percentages correctly guessed for each digit
#We also want to make sure that we correctly count the number of digits used for each different dataset



print('trouble-shooting')
print(guesses_for_test_data.shape)
test_set_size = len(training_data_0_4) ## CHANGE THIS WHEN YOU WANT TO USE A DIFFERENT TEST SET
test_labels = (training_labels_0_4) ## CHANGE THIS WHEN YOU WANT TO USE A DIFFERENT TEST SET


total_digit_count_test_set = np.zeros((10,1))
for i in range (10):
	print(guesses_for_test_data[i])
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
print("Percentages each number was guessed correctly using a neural network")
print(scaling_down_neural.shape)
print(scaling_down_neural)


################## TO DO###########
#Fix Back_Prop
# 
