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


def cost(inputs,outputs,hypothesis,total):
	first = -1.0*outputs*math.log(hypothesis)
	second = (1.0 - outputs)
	third = math.log(1.0 - hypothesis)
	cost = np.sum(first - (np.dot(second,third,)))
	return(cost/float(total))

def grad(hypothesis,inputs,outputs,total):
	second = hypothesis - outputs
	first = inputs.T
	grad = np.dot(first,second)
	return(grad/float(total))
def reg_cost(theta,l,outputs,inputs):
	total = 5000
	hypothesis = sigmoid(theta,inputs)
	first = -1.0*outputs*np.log(hypothesis)
	second = (1.0 - outputs)
	third = math.log(1.0 - hypothesis)
	cost = np.sum(first - (np.dot(second,third,))) #This is the normal cost function
	reg_cost = np.sum(np.dot(theta,theta.T) - np.dot(theta[0],theta[0].T))
	reg_cost = (l/float(2*total))*reg_cost
	return(cost+reg_cost)
def reg_grad(theta,l,outputs,inputs):
	total = 5000
	hypothesis = sigmoid(theta,inputs)
	second = hypothesis - outputs
	first = inputs
	grad = np.dot(first.T,second)
	print(grad.shape)
	print(theta.shape)
	print((l/float(total)))
	reg_grad = grad + (l/float(total))*theta
	reg_grad[0] = reg_grad[0] - (l/float(total))*theta[0] #Takes into account the fact that we want to leave theta_0 untouched (unregularized)
	return(reg_grad)

def display_Data(xvals):
	#This set of code generators each image (corresponding to 0-9)
	pic0 = np.transpose(np.reshape(xvals[0],(20,20)))
	print(pic0.shape)
	print(pic0)
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
l = 1.0 #Lambda
print(xvals.shape)
print(yvals.shape)
#Now let's add our intercepts to our xvals matrix
x_ones = np.ones((len(xvals),1),float)
xvals_intercepts = np.hstack((x_ones,xvals))
print(xvals_intercepts.shape)
#Next let's create our theta and temp_theta vectors
theta = np.zeros((401,10),float)
#theta and temp_theta and 401x10 matrices because we have 401 classes of theta and 10 different classifications
print(theta.shape)
display_Data(xvals)



#Now let's implement one vs all Classification 
for i in range(10): 	#We have 10 different classes 
	temp_theta = np.zeros(401,float)
	solution = np.zeros((5000,1),float)	#Next we will create an array where each individual element with a 1 will correspond to the same index in our yvals vector that has the number we are targeting 
	test = [10,1,2,3,4,5,6,7,8,9]		#Because MatLab is weird and indexes at 1, the 0 digits is labeled as 10 so we need to change this
	for k in range(total):
		
		if yvals[k] == test[i]:
			solution[i] = 1
		else: 
			solution[i] = 0 
						#Individually specifies which index is our "1" (target) and set the rest to "0's" 
	print(theta[i].shape)
	small =scipy.optimize.minimize(fun=reg_cost,x0=temp_theta,method='CG',jac=reg_grad,args = (l,solution,xvals_intercepts)) #This should find the theta values for each class (0-9) such that our regularized cost function is minimized
	theta[i,:] = small.x
	#Now let's plug each theta[i] into our sigmoid function and find each row that has the maximum probability
	sigmoid = sigmoid(theta[i],xvals_intercepts)
	print('sigmoid')
	print(sigmoid.shape)

#Next we wnat to find the highest value in each column of our hypothesis matrix which was originally (5000,10) and collapse it into a 1 x 5000 array with highest value of each training set
	sigmoid = np.reshape(sigmoid,(len(sigmoid),1))
	
	#Now let's stack together these theta values generated after each iteration into a bigger array
	if (i == 0): 
		array = sigmoid
	else: 
		array = np.hstack((array,sigmoid)) #Adds on the set of probabilities generated
#Now we here the loop is ended
#The next for loop loops through all the values in each column of array, finds the highest value and then stores the index of that highest value in the high_value array.
high_value = np.zeros((len(array),1))
for g in range(len(high_value)):
	high_value[i][0] = np.argmax(array[i,:])


#Next we want to test the accuracy of our regression method in being able to identify a number correctly
	
	
