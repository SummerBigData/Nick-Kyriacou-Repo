# Import packages
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
#First let's define a set of global variables
global training_sets #5000 training samples
global features #400 features per sample (excluding x-int bias we add)
global l #Lambda constant 
global num_classes #Ten total classes
global iterations #Counts number of iterations for any specific process
global length_hidden #Length of hidden layer (without bias added)
global epsilon #Used in gradient check function
global test_set_size #Size of our test set
#The following will be all the definitions of the functions for this script

def read_ids(filename, n = None): #This function used to read MNIST dataset
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB',f.read(4))
		shape  = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(),dtype = np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
			return arr


def sigmoid(z):
	z = np.array(z,dtype = np.float128)
	sigmoid = 1.0/(1.0+np.exp(-1.0*z))
	#print('did this: sigmoid')
	sigmoid = np.array(sigmoid,dtype = np.float64)
	return(sigmoid)

def grad_sigmoid(z):
	hypo = sigmoid(z)
	grad = hypo * (1.0 - hypo)
	#print('did this: grad_sigmoid')
	return(grad) 

def back_prop(theta_all,inputs,outputs):

	# First we must unroll our thetas
	theta_1 = theta_all[0:(features+1)*length_hidden]
	theta_2 = theta_all[(features+1)*length_hidden:len(theta_all)]
	theta_1 = np.reshape(theta_1,(length_hidden,features+1))
	theta_2 = np.reshape(theta_2,(num_classes,length_hidden+1))

	a_3, a_2 = feed_forward(theta_all, inputs)
	Delta2 = 0
	Delta1 = 0
	
	for i in range(len(inputs)):
		#First find the difference between outputs and output layer
		#Afterwards go ahead and compute each error term
		delta3 = a_3[i] - outputs[i]                   # Vector length 10
		
		delta3 = np.reshape(delta3, (len(delta3), 1))     # (10, 1) matrix
		temp_a2 = np.reshape(a_2[i], (len(a_2[i]), 1))      # (26, 1) matrix
		delta2 = np.multiply(np.dot(theta_2.T, delta3), temp_a2 * (1 - temp_a2))

		Delta2 = Delta2 + np.dot(delta3, temp_a2.T)       # (10, 26) matrix
		temp_inputs = np.reshape(inputs[i], (len(inputs[i]), 1))
		
		# We need to remove delta2[0] from Delta1
		Delta1 = Delta1 + np.delete(np.dot(delta2, temp_inputs.T), 0, axis = 0) # (25, 785) matrix are the correct dimensions (length_hiddenxfeatures+1)
		
	# Compute the unregularized gradient
	D1_unreg = (1.0 / float(training_sets)) * Delta1 
	D2_unreg = (1.0 / float(training_sets)) * Delta2
	
	# Compute the regularized gradient
	D1 = (1.0 / float(training_sets)) * Delta1 + (l / float(training_sets)) * theta_1
	D1[0:, 0:1] = D1_unreg[0:, 0:1]    # This makes it a (25, 785) matrix
	D2 = (1.0 / float(training_sets)) * Delta2 + (l / float(training_sets)) * theta_2
	D2[0:, 0:1] = D2_unreg[0:, 0:1]    # This makes it a (10, 26) matrix

	# Changed the gradient into a one dimensional vector
	D1 = np.ravel(D1)
	D2 = np.ravel(D2)
	D_unrolled = np.concatenate((D1, D2))

	return(D_unrolled)
		
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
	#This implements the unregularized cost function
	first = -1.0*np.multiply(outputs,np.log(hypothesis))
	second = np.multiply((1.0 - outputs), np.log(1 - hypothesis))
	cost_function = (1.0/float(training_sets))*np.sum((first - second))
	#Now to implement the regularized portion
	J = (1.0/float(training_sets))*np.sum(   -1*np.multiply(outputs, np.log(hypothesis)) - np.multiply(1-outputs, np.log(1 - hypothesis)	) )
	J_reg =  (0.5 * l / float(training_sets)) * ( np.sum(theta1**2) - np.sum(theta1[:,0]**2) + np.sum(theta2**2) - np.sum(theta2[:,0]**2) )
	return(J_reg + J)

	
def feed_forward(theta_all,xvals):
	#First we must unravel our theta1 and theta2 and reshape them into the correct dimensions
	theta1 = theta_all[0:(features+1)*length_hidden]
	theta2 = theta_all[(features+1)*length_hidden:len(theta_all)]
	theta1 = np.reshape(theta1,(length_hidden,features+1))
	theta2 = np.reshape(theta2,(num_classes,length_hidden+1))
	a_2 = sigmoid(np.dot(xvals,theta1.T))
	#print(a_2.shape)
	#print('that was a_2')
	ones = np.ones((len(xvals),1))
	a_2_bias = np.hstack((ones,a_2))
	#print('added in bias')
	#print(a_2_bias.shape)
	a_3 = sigmoid(np.dot(a_2_bias,theta2.T))
	#print(a_3.shape)
	#print('did this:feed_forward')
	return(a_3,a_2_bias)
	
def display_Data(xvals):
	#This set of code generates each image (corresponding to a digit 0:9)
	#FIXME
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
	return(y_array)

#Next let's load up our data from the MNIST DATABASE
training_sets = 20000 #1000 training samples
features = 784 #784 features per sample (excluding x-int bias we add) because we have set of 28x28 pixellated images
l = 1 #Lambda constant
lambda_mat = [1000,10,1,.1,.01] #matrix of lambda values
num_classes = 10
length_hidden = 25
test_set_size = 10000 #test set has 10000 entries
epsilon = .12
training_data = read_ids('train-images-idx3-ubyte.gz',training_sets)
training_labels = read_ids('train-labels-idx1-ubyte.gz',training_sets)
percentage_placeholder= np.zeros((len(lambda_mat),num_classes))
training_data = training_data/255.0

total_digit_count = np.zeros((10,1))
#Let's look at each element of training_labels and figure out how many total we have of each digit
for i in range(training_sets):
	digit = training_labels[i] 
	total_digit_count[digit] = total_digit_count[digit] + 1

print(total_digit_count)
#Now let's resize the data into usable form
training_data = np.reshape(training_data,(training_data.shape[0],training_data.shape[1]*training_data.shape[1]))
training_labels = np.reshape(training_labels,(training_sets,1))
print(training_data.shape)
print(training_labels.shape)
#Now add on bias element to training_data
x_ones = np.ones((training_sets,1))
training_data_int = np.hstack((x_ones,training_data))
print(training_data_int.shape)
#Now let's make our randomized Theta's
theta_random_1 = np.random.rand(length_hidden,(features+1))*2*epsilon - epsilon
theta_random_2 = np.random.rand((length_hidden+1),num_classes)*2*epsilon - epsilon
theta_random_1 = (theta_random_1)
theta_random_2 = (theta_random_2).T
theta_random_unrolled = np.concatenate((np.ravel(theta_random_1),np.ravel(theta_random_2)))
print('Shapes of thetas')
print(theta_random_1.shape)
print(theta_random_2.shape)
print(theta_random_unrolled.shape)
#We also need to record the labels for yvals as a vector containing only values 0 or 1 
y_mat = y_as_matrix(training_labels,training_sets)
print(y_mat.shape)
print(y_mat)


####################################THIS SECTION OF CODE DOES SOME MANIPULATION AND FORMATTING OF TEST SET TO PRACTICE ON##############################

#First we should import our test set data
test_data = read_ids('t10k-images-idx3-ubyte.gz',test_set_size)
test_labels = read_ids('t10k-labels-idx1-ubyte.gz',test_set_size)
test_data = test_data/float(255)
#Reshaping our training_data into 10000x784
test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[1]*test_data.shape[1]))
test_labels = np.reshape(test_labels,(test_set_size,1))

#Now let us take the optimal thetas obtained through using the best lambda value and feed_forward these values using the new test_data inputs to generate our hypothesis

x_ones = np.ones((test_set_size,1))
print(x_ones.shape)
print(test_data.shape)
test_data_int = np.hstack((x_ones,test_data)) 

#guesses_for_test_data, a_2_bias_test_data = feed_forward(optimal_thetas,test_data_int)
total_digit_count_test_set = np.zeros((num_classes,1))
#Let's look at each element of test_labels and figure out how many total we have of each digit
for i in range(test_set_size):
	digit = test_labels[i] 
	total_digit_count_test_set[digit] = total_digit_count_test_set[digit] + 1




#########################THIS SECTION CREATES BAR GRAPH FOR DIFFERENT LAMBDA PERCENTAGES################################

for i in range(len(lambda_mat)):
	l = lambda_mat[i]
	print('Lambda is ', l)
	print('now we are optimizing the cost function')
	optimize_time = scipy.optimize.minimize(fun = cost_function_reg,x0 = theta_random_unrolled,method = 'CG',tol = 1e-4,jac = back_prop,args = (training_data_int,y_mat))
	best_thetas = optimize_time.x
	best_thetas_cost = cost_function_reg(best_thetas,training_data_int,y_mat)
	print('Lowest cost function for lambda value of ',l ,'is ' , best_thetas_cost, 'training size was ' , training_sets)




	#Next we can find the percentages correctly guessed for each digit
	#First recover theta1 and theta2 values for each matrix
	
	guesses_for_test_data, a_2_bias_test_data = feed_forward(best_thetas,test_data_int)





	#Repeat the same process as above and find the index of the highest values and then store them in a test_set_sizex1 array
	highest_value_neural_network = np.zeros((test_set_size,1))

	for k in range(test_set_size):
		highest_value_neural_network[k,0] = np.argmax(guesses_for_test_data[k,:])

	#Now let's find the percentages we are able to identify a number correctly
	correctly_guessing_number_neural = np.zeros((10,1))




	for y in range(len(highest_value_neural_network)):
		if (highest_value_neural_network[y] == test_labels[y]):
			correctly_guessing_number_neural[test_labels[y]] = correctly_guessing_number_neural[test_labels[y]] + 1
	#Adding in some scaling factor to turn counts into percentages 
	scaling_down_neural = (correctly_guessing_number_neural)
	for s in range(num_classes):
		scaling_down_neural[s] = 100.0*(scaling_down_neural[s] / float(total_digit_count_test_set[s]) )
	print("Percentages each number was guessed correctly using a neural network")
	print(scaling_down_neural.shape)
	print(scaling_down_neural)
	#percentage_placeholder[i] = np.reshape(percentage_placeholder[i],(num_classes,1))
	print(percentage_placeholder[i])
	print(percentage_placeholder[i].shape)
	percentage_placeholder[i] = scaling_down_neural.flatten()

	#Before going back to the top of the for loop, now that we have found the best theta for 1 given lambda we must re-initialize a new set of theta values for our next lambda trial
	theta_random_1 = np.random.rand(length_hidden,(features+1))*2*epsilon - epsilon
	theta_random_2 = np.random.rand((length_hidden+1),num_classes)*2*epsilon - epsilon
	theta_random_1 = (theta_random_1)
	theta_random_2 = (theta_random_2).T
	theta_random_unrolled = np.concatenate((np.ravel(theta_random_1),np.ravel(theta_random_2)))
	
#Now let's plot a bar chart showing the Percentages guessed correctly for each set of lambda values and sample size used

print('testing')
print(scaling_down_neural[0].shape)
print(scaling_down_neural[0])
print(scaling_down_neural)
print(scaling_down_neural.shape)
x = [0, 1, 2 ,3 ,4 ,5 ,6 ,7 , 8 ,9]
x = np.reshape(x, (len(x), 1))
title = 'Accuracy of Different Lambda Values for Training size of '+ str(training_sets)

fig1 = plt.figure()
plt.bar(x - 0.35, percentage_placeholder[0], width = 0.14, align = 'center', color = 'goldenrod')
plt.bar(x - 0.21, percentage_placeholder[1], width = 0.14, align = 'center', color = 'blue')
plt.bar(x - 0.07, percentage_placeholder[2], width = 0.14, align = 'center', color = 'green')
plt.bar(x + 0.07, percentage_placeholder[3], width = 0.14, align = 'center', color = 'black')
plt.bar(x + 0.21, percentage_placeholder[4], width = 0.14, align = 'center', color = 'purple')
plt.title(title)
plt.xlim(left = -0.5, right = 9.5)
#plt.ylim(70)
plt.xlabel("Digit")
plt.ylabel('Percentage of correct guesses')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.legend(lambda_mat, loc = 'upper right', ncol = 6, prop = {'size': 9})
#plt.show()
plt.savefig('Different_Lambdas_20000_training_size_25_length_hidden.png')



######################DIFFERENT SECTION OF CODE#############################
'''
#Let use this snippet as a way to optimize best thetas for only 1 value of lambda used
print('Lambda is ', l)
print('now we are optimizing the cost function')
optimize_time = scipy.optimize.minimize(fun = cost_function_reg,x0 = theta_random_unrolled,method = 'CG',tol = 1e-4,jac = back_prop,args = (training_data_int,y_mat))
optimal_thetas = optimize_time.x
optimal_thetas_cost = cost_function_reg(optimal_thetas,training_data_int,y_mat)
print('Lowest cost function for lambda value of ',l ,'is ' , optimal_thetas_cost, 'training size was ' , training_sets)




#Next let us do some testing on our test set
#First we should import our test set data
test_data = read_ids('t10k-images-idx3-ubyte.gz',test_set_size)
test_labels = read_ids('t10k-labels-idx1-ubyte.gz',test_set_size)
test_data = test_data/float(255)
#Reshaping our training_data into 10000x784
test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[1]*test_data.shape[1]))
test_labels = np.reshape(test_labels,(test_set_size,1))

#Now let us take the optimal thetas obtained through using the best lambda value and feed_forward these values using the new test_data inputs to generate our hypothesis

x_ones = np.ones((test_set_size,1))
print(x_ones.shape)
print(test_data.shape)
test_data_int = np.hstack((x_ones,test_data)) 

guesses_for_test_data, a_2_bias_test_data = feed_forward(optimal_thetas,test_data_int)
total_digit_count_test_set = np.zeros((num_classes,1))
#Let's look at each element of test_labels and figure out how many total we have of each digit
for i in range(test_set_size):
	digit = test_labels[i] 
	total_digit_count_test_set[digit] = total_digit_count_test_set[digit] + 1


#Now that we have our best guesses for the test data let's compare them to our actual values to see how well we did
#This next step finds the index of the highest hypothesis value for each data entry in our test set and stores them in an array

highest_value_neural_network = np.zeros((test_set_size,1))

for k in range(test_set_size):
	highest_value_neural_network[k,0] = np.argmax(guesses_for_test_data[k,:])

	#Now let's find the percentages we are able to identify a number correctly
	correctly_guessing_number_neural = np.zeros((10,1))


	#Loop through all entries of the test_set and add one to the appropriate index of the array if it correctly guesses a test set entry
for y in range(test_set_size):
	if (highest_value_neural_network[y] == test_labels[y]):
		correctly_guessing_number_neural[test_labels[y]] = correctly_guessing_number_neural[test_labels[y]] + 1


#Finally let us calculate the percentages each number was guessed correctly
#Adding in some scaling factor to turn counts into percentages (500 divided by because 500 of each number was inputed)
scaling_down_neural = (correctly_guessing_number_neural)
for s in range(num_classes):
	scaling_down_neural[s] = 100.0*(scaling_down_neural[s] / float(total_digit_count_test_set[s]) )
print("Percentages each number of test set (size 10,000) was guessed correctly using a neural network")
print(scaling_down_neural)


#This reads output to a different file
output_file_name = 'training_set_' + str(training_sets) + '_lambda_'+ str(l) + '_hidden_layer_' +str(length_hidden) + '_test_set_' + str(test_set_size) + '.out'
#F = open(output_file_name,"r+")
#F.write("This run used " + str(training_sets) + " training sameples with a lambda of " + string(l) + " to train a neural network of 3 layers, 10 output classes and " + str(length_hidden) + " units in the hidden layer. After optimizing thetas these values were test on a test set size of " + str(test_set_size) + " And the following percentages each number of test set were guessed correctly were: ")
#F.write(scaling_down_neural)

np.savetxt(output_file_name,scaling_down_neural,delimiter=',')

'''

###################THIS IS OLD-REHASHED CODE#############################

#Now let's go into backpropagation
#print('we are going into the back prop')
#back_prop(theta_random_unrolled,training_data_int,y_mat)
#Now let's check our gradient
#print('we are checking our grad')
#print(scipy.optimize.check_grad(cost_function_reg,back_prop,theta_random_unrolled,training_data_int,y_mat))
#The check_grad did quite well, recieved 0.0164 thus we can ignore it for future coding. 
#Next let's find the theta values that give us the optimized cost function
#for k in range(len(lambda_mat)):
#	l = lambda_mat[k]
#	print('Lambda is ', l)

#best_thetas_with_lambdas = np.zeros((best_thetas,len(lambda_mat)))
#best_thetas_with_lambda[k] = best_thetas
#print('It took ', iterations, 'iterations to minimize the cost function')
#Now let's plug these into the cost function to see what our lowest value of the cost function is
#best_thetas_cost = np.zeros((lambda_mat,num_classes))
#for k in range(len(lambda_mat)):
#best_thetas_cost[k] = cost_function_reg(best_thetas_with_lambdas[k],training_data_int,y_mat)
#print('Lowest cost function for lambda value of ',lambda_mat[k], 'is ' , best_thetas_cost[k])





#print(best_guess[0,:])
#print(best_guess[1,:])
#print(best_guess[2,:])
#print(best_guess[3,:])
#print(best_guess[4,:])
#print(highest_value_neural_network[0:20])
#print('training labels')
#print(training_labels[0:20])

#highest_value_neural_network = np.roll(highest_value_neural_network,4500)



#Let's make a bar graph of the performance of our learning algorithm for different values of lambda

'''
a_3,a_2_bias  = feed_forward(theta_all,inputs)
Triangle_layer_1 = np.zeros((length_hidden,features+1)) #Creates the 25x401 Triangle matrix for the 1st layer 
#Triangle_layer_1 = 0
Triangle_layer_2 = np.zeros((num_classes,length_hidden+1)) # Creates the 10 x 26 Triangle matrix for the 2nd layer
#Triangle_layer_2 = 0
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
	print "shapes Triangle_layer_1 ",Triangle_layer_1.shape
	print "shapes Triangle_layer_2 ",Triangle_layer_2.shape
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
print('did this: back prop')
'''

#This was my cost function


'''
cost = cost_function(theta_all,inputs,outputs)
theta1_temp = (np.dot(theta1,theta1.T) - theta1[0,0]**2)
theta2_temp = (np.dot(theta2,theta2.T) - theta2[0,0]**2)
thetas = (l/(float(2*training_sets)))* (np.sum(theta1_temp) + np.sum(theta2_temp))	
cost_function_reg = thetas+cost
#print('did this:cost_func_reg: ', cost_function_reg)

'''
