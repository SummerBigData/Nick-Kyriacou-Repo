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

#The following will be all the definitions of the functions for this script
def sigmoid(theta,xvals):
	z = np.dot(xvals,theta.T)
	sigmoid = 1.0/(1.0+np.exp(z))
	return(sigmoid)

def cost_function(outputs,hypothesis,m):
	print(outputs.shape)
	print(hypothesis.shape)
	first = -1.0*np.multiply(outputs,np.log(hypothesis))
	second = np.multiply((1.0 - outputs), np.log(1 - hypothesis))
	cost_function = (1.0/float(m))*np.sum((first - second))
	return(cost_function)
	
def feed_forward(theta1,theta2,xvals):
	a_2 = sigmoid(theta1,xvals)
	ones = np.ones((len(a_2),1))
	a_2_bias = np.hstack((ones,a_2))
	a_3 = sigmoid(theta2,a_2_bias)
	print(a_3)	
	return(a_3)
	
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
x_ones = np.ones((len(xvals),1))
xvals_ones = np.hstack((x_ones,xvals))
#We also need to record the labels for yvals as a vector containing only values 0 or 1 
y_mat = y_as_matrix(yvals,training_sets)
theta_1 = weights['Theta1']
print(theta_1.shape)
theta_2 = weights['Theta2']
print(theta_2.shape)
#display_Data(xvals)
outputs = feed_forward(theta_1,theta_2,xvals_ones)
cost = cost_function(y_mat,outputs,training_sets)
print(cost)

