#Purpose: The purpose of this code is to visualize the activation units of the hidden layer after running it through the sparse auto encoder to determine structure in the data-set
#Created by: Nick Kyriacou
#Created on: 6/13/2018

# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.image as mpimg
from scipy.optimize import minimize 
import struct as st
import gzip

####### Global variables #######
global training_sets
global features
global length_hidden 
global l  #Lambda Parameter
global Beta #Beta Parameter
global P  #Rho Parameter
####### Function Definitions #######


################ DATA PROCESSING FUNCTIONS #####################
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

def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	normed = ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin
	return(normed)
############################# OTHER FUNCTIONS GO HERE #######################3
def seperate(theta_all): #This function will take a combined theta vector and seperate it into 4 of its specific components
	print('in seperate')
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features))
	print(W_1.shape)
	B_1 = np.reshape(theta_all[2*features*length_hidden:2*features*length_hidden + length_hidden],(length_hidden,1))
	print(B_1.shape)
	W_2 = np.reshape(theta_all[features*length_hidden:2*features*length_hidden],(features,length_hidden))
	print(W_2.shape)
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1))
	print(B_2.shape)
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
################### MAIN CODE STARTS HERE #############################3

#First let's load our optimal weights
training_sets = 60000
features  = 784
length_hidden = 200
l = 3e-7
Beta = 3
P = 0.1
file_name = 'output_folder/Best_Weights_Rho' + str(P) + 'Lambda' + str(l) + 'Beta' + str(Beta) + '.out' #This is the name of the file where we pull our theta weights from
optimal_thetas = np.genfromtxt(file_name,dtype = 'float')
print(optimal_thetas.shape)


#Next let's get our randomized data-set of 5-9 inputs
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


#Now let's feed forward to find the best guesses and also unravel our best thetas to get the weights for each unit. 
a_3_best, a_2_best = feed_forward(optimal_thetas, training_data_5_9) 
print(a_3_best.shape)
print(a_2_best.shape)
W_1, B_1, W_2, B_2 = seperate(optimal_thetas)


denominator = np.sum(W_1**2)**(-0.5)
X = W_1 / denominator			
#X = Norm(X)
#print(X.shape)
#print(X)

#Now let us display each of these images. There should be 200 28x28 pixel images.
input_pic = np.zeros((length_hidden,28,28))
for i in range(196):
	#seperate each picture from 784 features to 28x28 pixel
	all_pixels = np.ravel(X[i])
	input_pic[i] = np.reshape(all_pixels,(28,28))

#Maybe create some black space I guess
maximum = X.max()
black_space_horizontal = np.ones((2,422))*maximum # 434 determined by 14*28 (one for each pixel) + 14*1 one for each vertical seperation + 28*1 one for the vertical seperation at the beginning then divide 434/14 to find each spacing size
black_space_vertical = np.ones((28,2))*maximum

#black_space_horizontal = np.asarray([ [1 for i in range(52)] for j in range(2)])

Combined_panel = black_space_horizontal #Set the first element of each picture to be some black space

#Next let's concatenate these images
for i in range(14):
	#print(input_pic[i].shape)
	row = np.concatenate((black_space_vertical,input_pic[i*14],black_space_vertical,input_pic[i*14+1],black_space_vertical,input_pic[i*14+2],black_space_vertical,input_pic[i*14+3],black_space_vertical,input_pic[i*14+4],black_space_vertical,input_pic[i*14+5],black_space_vertical,input_pic[i*14+6],black_space_vertical,input_pic[i*14+7],black_space_vertical,input_pic[i*14+8],black_space_vertical,input_pic[i*14+9],black_space_vertical,input_pic[i*14+10],black_space_vertical,input_pic[i*14+11],black_space_vertical,input_pic[i*14+12],black_space_vertical,input_pic[i*14+13],black_space_vertical),axis = 1)
	#print(Combined_panel.shape)
	#print(row.shape)
	Combined_panel = np.vstack((Combined_panel,row,black_space_horizontal))

imgplot = plt.imshow(Combined_panel, cmap="binary", interpolation='none') 
plt.savefig('Picture_Folder/Visualizing_Activations_for_hidden_layer_node_Lamb_'+str(l)+'_Beta_'+str(Beta) +'_Rho_'+str(P) +'.png',transparent=False, format='png')
plt.show()
