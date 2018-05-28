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

def sigmoid(theta,inputs):
	z = np.dot(inputs,theta)
	sigmoid = 1.0/(1.0+np.exp(-1.0*z))
	return(sigmoid)

def reg_cost(theta,l,outputs,inputs):
	theta = np.reshape(theta,(1,len(theta))) #Resizes theta as a (401,1) array instead of a (401,) list
	total = 5000
	hypothesis = sigmoid(theta.T,inputs)
	first = np.dot(outputs.T,np.log(hypothesis))
	second = np.dot((1.0 - outputs).T,np.log(1-hypothesis))
	reg_cost = (-1.0/float(total))*(first +second) + (l/(2.0*float(total)))*(np.dot(theta,theta.T) - theta[0,0]**2)
	return np.asscalar(reg_cost)
def reg_grad(theta,l,outputs,inputs):
	theta = np.reshape(theta,(1,len(theta)))  #Resizes theta as a (401,1) array instead of a (401,) list
	total = 5000
	hypothesis = sigmoid(theta.T,inputs)
	first = (1.0/float(total))*np.dot((hypothesis - outputs).T,inputs)
	reg_grad = (1.0/float(total))*np.add(np.dot((hypothesis - outputs).T,inputs),(l/(float(total)))*theta)
	reg_grad[0,0] = first[0,0] #Takes into account the fact that we want to leave theta_0 untouched (unregularized)
	return(reg_grad.flatten())

def display_Data(xvals):
	#This set of code generators each image (corresponding to 0-9)
	pic0 = np.transpose(np.reshape(xvals[0],(20,20)))
	pic1 = np.transpose(np.reshape(xvals[501],(20,20)))
	pic2 = np.transpose(np.reshape(xvals[1001],(20,20)))
	pic3 = np.transpose(np.reshape(xvals[1501],(20,20)))
	pic4 = np.transpose(np.reshape(xvals[2001],(20,20)))
	pic5 = np.transpose(np.reshape(xvals[2501],(20,20)))
	pic6 = np.transpose(np.reshape(xvals[3001],(20,20)))
	pic7 = np.transpose(np.reshape(xvals[3501],(20,20)))
	pic8 = np.transpose(np.reshape(xvals[4001],(20,20)))
	pic9 = np.transpose(np.reshape(xvals[4501],(20,20)))
	#Now we can combined them all together to be show on a singular panel
	pic_combined = np.concatenate((pic0,pic1,pic2,pic3,pic4,pic5,pic6,pic7,pic8,pic9),axis = 1)
	img_plot = plt.imshow(pic_combined,cmap = 'binary')
	
	plt.show()
	

#Loading the Data
data = scipy.io.loadmat('ex3data1.mat')
xvals = data['X']
yvals = data['y']
total = 5000 #5000 training samples
features = 400 #400 features (excluding the x-intercept)
alpha = 1.0 #Learning rate
l = 1000 #Lambda

#Now let's add our intercepts to our xvals matrix
x_ones = np.ones((len(xvals),1),float)
xvals_intercepts = np.hstack((x_ones,xvals))

#Next let's create our theta and temp_theta vectors
theta = np.zeros((401,10),float)
#theta and temp_theta and 401x10 matrices because we have 401 classes of theta and 10 different classifications

display_Data(xvals)



#Now let's implement one vs all Classification 
for i in range(10): 	#We have 10 different classes 
	temp_theta = np.zeros((1,401),float)
	solution = np.zeros((5000,1))	#Next we will create an array where each individual element with a 1 will correspond to the same index in our yvals vector that has the number we are targeting 
	test = [10,1,2,3,4,5,6,7,8,9]		#Because MatLab is weird and indexes at 1, the 0 digits is labeled as 10 so we need to change this
	for k in range(total):
		
		if (yvals[k] == test[i]):
			solution[k] = 1
		else: 
			solution[k] = 0 
						#Individually specifies which index is our "1" (target) and set the rest to "0's" 
	
	small = scipy.optimize.minimize(fun=reg_cost,x0=temp_theta,method='CG',jac=reg_grad,args = (l,solution,xvals_intercepts)) #This should find the theta values for each class (0-9) such that our regularized cost function is minimized
	temp_theta = np.reshape(small.x,(1,len(small.x)))
	
	#Now let's plug each theta[i] into our sigmoid function and find each row that has the maximum probability
	guess = sigmoid(temp_theta.T,xvals_intercepts)
#Next we wnat to find the highest value in each column of our hypothesis matrix which was originally (5000,10) and collapse it into a 1 x 5000 array with highest value of each training set

	guess = np.reshape(guess,(len(guess),1))
	#Now let's stack together these theta values generated after each iteration into a bigger array
	if (i == 0): 
		array = guess
	else: 
		array = np.hstack((array,guess)) #Adds on the set of probabilities generated
	print(i)
#Now we here the loop is ended
#The next for loop loops through all the values in each column of array, finds the highest value and then stores the index of that highest value in the high_value array.
high_value = np.zeros((len(array),1))
for g in range(len(array)):
	high_value[g,0] = np.argmax(array[g,:])

#Next we want to test the accuracy of our regression method in being able to identify a number correctly
correctly_guessing_number = np.zeros((10,1))
for h in range(len(high_value)):
	if (high_value[h] == yvals[h]): #In this if statement, if we correctly guess the output then we add one to the count of correctly guessed values 
		correctly_guessing_number[yvals[h]] = correctly_guessing_number[yvals[h]]+1
	if (high_value[h] == 0 and yvals[h] ==10): #Takes into account the fact that indexing for number 0 started at 10 with Matlab
		correctly_guessing_number[0] = correctly_guessing_number[0] + 1
scaling_down = (correctly_guessing_number/float(500.0))*100
print('Percentage each number was guessed correctly using lambda ' ,l)
print(scaling_down)


#Now it is time that we build some neural networks!
#We must first import the set of weights already given to us
print("Now we are building a neural network")
weights_data = scipy.io.loadmat('ex3weights.mat')
Theta_1 = weights_data['Theta1']
Theta_2 = weights_data['Theta2']

#Theta_1 is the matrix of weights for the first layer
#Theta_2 is the matrix of weights for the second (hidden) layer
#We have three layers, let us first compute the activation of unit i, [a] in the second layer
a_2 =  sigmoid(Theta_1.T, xvals_intercepts)


#Next we need to add the bias layer onto a_2 
a_2 = np.hstack((x_ones, a_2))


#Now let's run the (Theta) values from the second (hidden) layer into our hypothesis function
output = sigmoid(Theta_2.T, a_2)

#Repeat the same process as above and find the index of the highest values and then store them in a 5000x1 array
highest_value_neural_network = np.zeros((total,1))

for k in range(total):
	highest_value_neural_network[k,0] = np.argmax(output[k,:])


highest_value_neural_network = np.roll(highest_value_neural_network,4500)
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
	
