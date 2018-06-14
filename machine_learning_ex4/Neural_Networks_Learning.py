# Import packages
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
import scipy.io
import matplotlib.image as mpimg

from scipy.optimize import minimize 
#First let's define a set of global variables
global training_sets #5000 training samples
global features #400 features per sample (excluding x-int bias we add)
global l #Lambda constant 
global num_classes #Ten total classes
global iterations #Counts number of iterations for any specific process
global length_hidden #Length of hidden layer (without bias added)
global epsilon #Used to size parameters of randomly generated thetas
#The following will be all the definitions of the functions for this script


def sigmoid(z):
	sigmoid = 1.0/(1.0+np.exp(-1.0*z))
	#print('did this: sigmoid')
	return(sigmoid)

def grad_sigmoid(z):
	hypo = sigmoid(z)
	grad = hypo * (1.0 - hypo)
	#print('did this: grad_sigmoid')
	return(grad) 

def back_prop(theta_all,inputs,outputs):
	#First we must unravel our theta1 and theta2 and reshape them into the correct dimensions
	#theta1 = theta_all[0:10025]
	#theta2 = theta_all[10025:len(theta_all)]
	theta1 = np.reshape(theta_all[0:10025],(25,401))
	theta2 = np.reshape(theta_all[10025:len(theta_all)],(10,26))

	a_3,a_2_bias  = feed_forward(theta_all,inputs)
	#Triangle_layer_1 = np.zeros((length_hidden,features+1)) #Creates the 25x401 Triangle matrix for the 1st layer 
	Triangle_layer_1 = 0
	#Triangle_layer_2 = np.zeros((num_classes,length_hidden+1)) # Creates the 10 x 26 Triangle matrix for the 2nd layer
	Triangle_layer_2 = 0
	#Creates delta for output (third) layer
	# Now that we have defined some necessary variables let's begin looping through each training sample
	

	for i in range(training_sets):
		temp_a_2_bias = np.reshape(a_2_bias[i],(len(a_2_bias[i]),1))

		#Finds error term for each output unit in layer 3
		Delta_3 = a_3[i] - outputs[i] 
		#Next finds the error in hidden layer 2 by taking the dot product of the weights with error from layer 3 
		first = np.dot(theta2.T,Delta_3)

		first = np.reshape(first,(len(first),1))
		second = grad_sigmoid(temp_a_2_bias)
		Delta_2 = np.multiply(first , second)

		#Reprocesses Delta_2 into a usable form and gets rid of the bias layer as this is no longer necessary
		Delta_2 = Delta_2.reshape(len(Delta_2),1)
		Delta_2 = np.delete(Delta_2,0,0)

		#This next step accumulates the gradient
		#Isolates each individual training sample into an array
		inputs_temp = np.reshape(inputs[i],(len(inputs[i]),1))
		Triangle_layer_1 = Triangle_layer_1 + np.dot(Delta_2,inputs_temp.T)

		#Also reshapes Delta_3 into usable form
		Delta_3 = np.reshape(Delta_3,(len(Delta_3),1))
		Triangle_layer_2 = Triangle_layer_2 + np.dot(Delta_3,temp_a_2_bias.T)

	#Divides accumulated gradient by m to compute unregularized gradient
	D_1_unreg = Triangle_layer_1/float(training_sets)
	D_2_unreg = Triangle_layer_2/float(training_sets)

	#Now we can find the regularized gradient
	#For j>=1 we use the regularization value
	D_1_reg = Triangle_layer_1/float(training_sets) + (l/float(training_sets)) * theta1
	D_2_reg	= Triangle_layer_2/float(training_sets)	+ (l/float(training_sets)) * theta2
	#Note there are two different conditions, for j = 0 there is no regularization	
	D_1_reg[0:,0:1] = D_1_unreg[0:,0:1]
	D_2_reg[0:,0:1] = D_2_unreg[0:,0:1]
	
	#This can be changed depending on whether or not we want to return the regularized or unregularized gradients 
	D_unrolled = np.concatenate((np.ravel(D_1_reg),np.ravel(D_2_reg)))		
	#print('find D_1, D_2')
	#print(D_unrolled.shape)
	#print('did this: back prop')
	return(D_unrolled)
		
	#print(i)

def grad_check(theta_all,inputs,outputs):
	theta_upper = theta_all + epsilon
	theta1 = theta_upper[0:10025]
	theta1 = np.reshape(theta1,(25,401))
	theta2 = theta_upper[10025:]
	theta2 = np.reshape(theta2,(10,26))
	#Now re-roll them back up
	theta_unrolled_upper= np.concatenate((np.ravel(theta1),np.ravel(theta2)))
	cost_function_upper = cost_function_reg(theta_unrolled_upper,inputs,outputs)
	theta_lower = theta_all - epsilon
	theta1 = theta_lower[0:10025]
	theta2 = theta_lower[10025:len(theta_all)]
	theta1 = np.reshape(theta1,(25,401))
	theta2 = np.reshape(theta2,(10,26))
	#Now re-roll them back up
	theta_unrolled_lower= np.concatenate((np.ravel(theta1),np.ravel(theta2)))
	cost_function_lower = cost_function_reg(theta_unrolled_lower,inputs,outputs)
	grad_check = (cost_function_upper - cost_function_lower)/float(2*epsilon)
	return(grad_check)

def cost_function_reg(theta_all,inputs,outputs):
	#First we must unravel our theta1 and theta2 and reshape them into the correct dimensions
	theta1 = theta_all[0:10025]
	theta2 = theta_all[10025:len(theta_all)]
	theta1 = np.reshape(theta1,(25,401))
	theta2 = np.reshape(theta2,(10,26))
	hypothesis,a_2_bias = feed_forward(theta_all,inputs)

	cost = cost_function(theta_all,inputs,outputs)
	theta1_temp = (np.dot(theta1,theta1.T) - theta1[0,0]**2)
	theta2_temp = (np.dot(theta2,theta2.T) - theta2[0,0]**2)
	thetas = (l/(float(2*training_sets)))* (np.sum(theta1_temp) + np.sum(theta2_temp))	
	cost_function_reg = thetas+cost
	#print('did this:cost_func_reg')
	return(cost_function_reg)

def cost_function(theta_all,inputs,outputs):
	#print('inside cost function')
	#print(theta_all.shape)
	theta1 = theta_all[0:10025]
	theta2 = theta_all[10025:len(theta_all)]
	theta1 = np.reshape(theta1,(25,401))
	theta2 = np.reshape(theta2,(10,26))
	hypothesis,a_2_bias = feed_forward(theta_all,inputs)
	first = -1.0*np.multiply(outputs,np.log(hypothesis))
	second = np.multiply((1.0 - outputs), np.log(1 - hypothesis))
	cost_function = (1.0/float(training_sets))*np.sum((first - second))
	#print('did this: cost_func')
	return(cost_function)
	
def feed_forward(theta_all,xvals):
	#First we must unravel our theta1 and theta2 and reshape them into the correct dimensions
	theta1 = theta_all[0:10025]
	theta2 = theta_all[10025:len(theta_all)]
	theta1 = np.reshape(theta1,(25,401))
	theta2 = np.reshape(theta2,(10,26))
	a_2 = sigmoid(np.dot(xvals,theta1.T))
	#print(a_2.shape)
	#print('that was a_2')
	ones = np.ones((training_sets,1))
	a_2_bias = np.hstack((ones,a_2))
	#print('added in bias')
	#print(a_2_bias.shape)
	a_3 = sigmoid(np.dot(a_2_bias,theta2.T))
	#print(a_3.shape)
	#print('did this:feed_forward')
	return(a_3,a_2_bias)
	
def display_Data(xvals):
	#This set of code generates each image (corresponding to a digit 0:9)
	pic0 = np.transpose(np.reshape(xvals[0],(20,20)))
	pic1 = np.transpose(np.reshape(xvals[500],(20,20)))
	pic2 = np.transpose(np.reshape(xvals[1000],(20,20)))
	pic3 = np.transpose(np.reshape(xvals[1500],(20,20)))
	pic4 = np.transpose(np.reshape(xvals[2000],(20,20)))
	pic5 = np.transpose(np.reshape(xvals[2500],(20,20)))
	pic6 = np.transpose(np.reshape(xvals[3000],(20,20)))
	pic7 = np.transpose(np.reshape(xvals[3500],(20,20)))
	pic8 = np.transpose(np.reshape(xvals[4000],(20,20)))
	pic9 = np.transpose(np.reshape(xvals[4500],(20,20)))

	pic_combined = np.concatenate((pic0,pic1,pic2,pic3,pic4,pic5,pic6,pic7,pic8,pic9),axis = 1)
	img_plot = plt.imshow(pic_combined,cmap = 'binary')
	
	plt.show()

def y_as_matrix(y,training_sets): #This takes a 5000x1 vector and makes it into a 5000x10 matrix, Code based on one made by suren
	y = np.ravel(y)
	y_array = np.zeros((training_sets,10))
	for i in range(len(y)):
		for j in range(10):
			if (y[i] == j):
				y_array[i][j] = 1
			if (y[i] == 10 and j == 0):
				y_array[i][j] = 1
	return(y_array)

#Next let's load up our data from the exercise
data = scipy.io.loadmat('ex4data1.mat')
weights = scipy.io.loadmat('ex4weights.mat')
xvals = data['X']
yvals = data['y']
training_sets = len(xvals) #5000 training samples
features = len(xvals.T) #400 features per sample (excluding x-int bias we add)
l = 1 #Lambda constant 
num_classes = 10
length_hidden = 25
epsilon = .12

x_ones = np.ones((len(xvals),1))
xvals_ones = np.hstack((x_ones,xvals))
#We also need to record the labels for yvals as a vector containing only values 0 or 1 
y_mat = y_as_matrix(yvals,training_sets)
theta_1 = weights['Theta1']
theta_2 = weights['Theta2']

theta_unrolled = np.concatenate((np.ravel(theta_1),np.ravel(theta_2)))
print(theta_unrolled.shape) #this creates a vector of a list of theta's
print(y_mat)
#These are commented out because now feed_forward returns both hidden and output layer values
#display_Data(xvals)
#outputs,x = feed_forward(theta_1,theta_2,xvals_ones)
#cost = cost_function(y_mat,outputs,training_sets)
#print(cost)
#cost_regularized = cost_function_reg(theta_1,theta_2,xvals_ones,y_mat,training_sets,l)
#print(cost_regularized)

#print(grad_sigmoid(0))
#WAS JUST PLAYING WITH MAKING RANDOM VALUES FOR MY THETA, NOT IMPORTANT
print('length of theta_1 and theta_2', len(theta_1), len(theta_2.T))
theta_random_1 = np.random.rand(len(theta_1),features+1)*2*epsilon  - epsilon
theta_random_2 = np.random.rand(len(theta_2.T),10)*2*epsilon - epsilon
theta_random_1 = (theta_random_1)
theta_random_2 = (theta_random_2).T
#For some reason creating random theta matrices gives me a much higher cost so I don't want to look too heavily in that
#outputs = feed_forward(theta_random_1, theta_random_2, xvals_ones)
print('random')
theta_random_unrolled = np.concatenate((np.ravel(theta_random_1),np.ravel(theta_random_2)))
#print(cost_function(y_mat,outputs,training_sets))
#print(cost_function_reg(theta_random_1, theta_random_2, xvals_ones,y_mat,training_sets,l))



#Now let's go into backpropagation
back_prop(theta_random_unrolled,xvals_ones,y_mat)
#Now let's check our gradient
#print(scipy.optimize.check_grad(cost_function_reg,back_prop,theta_random_unrolled,xvals_ones,y_mat))
#Next let's find the theta values that give us the optimized cost function
optimize_time = scipy.optimize.minimize(fun = cost_function_reg,x0 = theta_random_unrolled,method = 'CG',tol = 1e-4,jac = back_prop,args = (xvals_ones,y_mat))
best_thetas = optimize_time.x
#Now let's plug these into the cost function to see what our lowest value of the cost function is
best_thetas_cost = cost_function_reg(best_thetas,xvals_ones,y_mat)
print('Lowest cost function ' , best_thetas_cost)

#Next we can find the percentages correctly guessed for each digit

#First recover theta1 and theta2 values for each matrix
theta1 = best_thetas[0:10025]
theta1 = np.reshape(theta1,(25,401))
theta2 = best_thetas[10025:len(best_thetas)]
theta2 = np.reshape(theta2,(10,26))
best_guess,a_2_bias = feed_forward(best_thetas, xvals_ones)

#Repeat the same process as above and find the index of the highest values and then store them in a 5000x1 array
highest_value_neural_network = np.zeros((training_sets,1))

for k in range(training_sets):
	highest_value_neural_network[k,0] = np.argmax(best_guess[k,:])


#highest_value_neural_network = np.roll(highest_value_neural_network,4500)
#Now let's find the percentages we are able to identify a number correctly
correctly_guessing_number_neural = np.zeros((10,1))

for y in range(len(highest_value_neural_network)):
	if (highest_value_neural_network[y] == yvals[y]):
		correctly_guessing_number_neural[yvals[y]] = correctly_guessing_number_neural[yvals[y]] + 1
	if (highest_value_neural_network[y] == 0 and yvals[y] == 10):
		correctly_guessing_number_neural[0] = correctly_guessing_number_neural[0] + 1
#Adding in some scaling factor to turn counts into percentages (500 divided by because 500 of each number was inputed)
scaling_down_neural = (correctly_guessing_number_neural)*100.0/float(500)
print("Percentages each number was guessed correctly using a neural network")
print(scaling_down_neural)






#digit = grad_check(theta_unrolled,xvals_ones,y_mat)
#print(digit)
