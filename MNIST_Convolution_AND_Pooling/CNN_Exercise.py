#Purpose: This is the main exercise code which calls the convolution and pooling functions.
#Created By: Nick Kyriacou
#Created on: 6/21/2018



################################# IMPORTING PACKAGES ########################


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io
from scipy.optimize import minimize 
import scipy.signal
from Convolution import Convolve
from Pooling import Pool
import random
import sys
import struct as st
import gzip

################################## GLOBAL CONSTANT DEFINITIONS #######################
training_input_size = 60000 #Number of MNIST samples
testing_size = 10000
features = 225
length_hidden = 100
patch_size = 15 # 15x15
imageDim = 28 #28x28 pixel images
poolDim = 7 #Dimension of Pooling region
STEP_SIZE = 25 # Convolve and Pool only 50 features at a time to not run out of memory
########################### FUNCTION DEFINITIONS #############################



def read_ids(filename, n = None): #This function used to read MNIST dataset
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB',f.read(4))
		shape  = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(),dtype = np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
			return arr



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


def sigmoid(z): 
	hypothesis = 1.0/(1.0 + np.exp(-z) )
	return(hypothesis)



def check_convolve(image,convolved_feature,W_1,B_1): #This code checks whether my convolution function did what it was supposed to (form borrowed from Matt)
	num_images = len(image)

	for i in range(1000):
		# Pick a random data patch
		feature_num = np.random.randint(0, length_hidden - 1)
		image_num = np.random.randint(0, num_images)
		image_row = np.random.randint(0, imageDim - patch_size)
		image_col = np.random.randint(0, imageDim - patch_size)
	
		patch = image[image_num,image_row: image_row + patch_size, image_col: image_col + patch_size]
		print(patch.shape)
		print('that was shape of patch')
		patch = patch.reshape(1,patch_size**2)
		
		
		features = sigmoid(np.dot(patch, W_1.T) + np.tile(np.ravel(B_1), (num_images, 1)))
		
		if (abs(features[0, feature_num] - convolved_feature[feature_num, image_num, image_row, image_col]) > 1e-9):
			print 'Convolved feature does not match activation from autoencoder'
			print 'Feature Number    : ', feature_num
			print 'Image Number      : ', image_num
			print 'Image Row         : ', image_row
			print 'Image Column      : ', image_col
			print 'Convolved feature : ', conv_feat[feature_num, image_num, image_row, image_col]
			print 'Sparse AE feature : ', features[1, feature_num]
			print 'Error: Convolved feature does not match activation from autoencoder'
			sys.exit()
		
	print 'Congratulations! Your convolution code passed the test.'


	return()


def check_pool(): #This code checks our pooling function (Form borrowed from Matt)

	# Set up a test matrix to test on
	test_matrix = np.arange(1, 65).reshape(8, 8)

	# Manually compute our results to check against
	expected_matrix = np.array((np.mean(np.mean(test_matrix[0:4, 0:4])), np.mean(np.mean(test_matrix[0:4, 4:8])), np.mean(np.mean(test_matrix[4:8, 0:4])), np.mean(np.mean(test_matrix[4:8, 4:8])))).reshape(2, 2)

	test_matrix = np.reshape(test_matrix, (1, 1, 8, 8))

	pool_feat = np.squeeze(Pool(4, test_matrix))

	if np.any(np.not_equal(pool_feat, expected_matrix) == True):
		print 'Pooling incorrect'
		print 'Expected'
		print expected_matrix
		print 'Got'
		print pool_feat
		sys.exit()
	else:
		print 'Congratulations! Your pooling code passed the test.'

	return
	

################################ MAIN CODE STARTS HERE  ###########################
l = 10 #FIXME Change based on the run we get
Beta = 0.5 #FIXME Change based on the run we get
P = 0.05 #FIXME Change based on the run we get

#First let's get all our data that was previously generated
input_name = 'output_folder/optimal_thetas_l_' + str(l) + '_B_' + str(Beta) + '_Rho_' + str(P) + '.out'
best_weights = np.genfromtxt(input_name,dtype = float)
#Next let's unpackage our theta weights
W_1_Best,B_1_Best,W_2_Best,B_2_Best = seperate(best_weights)

#Now let's grab our MNIST data
training_data = read_ids('data/train-images-idx3-ubyte.gz',training_input_size)
training_labels = read_ids('data/train-labels-idx1-ubyte.gz',training_input_size)
testing_data = read_ids('data/t10k-images-idx3-ubyte.gz',testing_size)
testing_labels = read_ids('data/t10k-labels-idx1-ubyte.gz',testing_size)

#Let's normalize the training and testing data
training_data = training_data/ 255.0
testing_data = testing_data /255.0
print(W_1_Best.shape)
print(B_1_Best.shape)
print(W_2_Best.shape)
print(B_2_Best.shape)


#Now that we have loaded in our data let us next take a look at creating our matrix we will use to pool. Dimensions of (length_hidden,num_samples,num_pooling_regions,num_pooling_regions)
pooling_training = np.zeros((length_hidden,training_input_size,(imageDim - patch_size + 1)/poolDim,(imageDim - patch_size + 1)/poolDim)) # (100,600000,7,7)
pooling_testing = np.zeros((length_hidden,testing_size,(imageDim - patch_size + 1)/poolDim,(imageDim - patch_size + 1)/poolDim))      # (100,10000,7,7)

print(pooling_training.shape)
print(pooling_testing.shape)

'''
#First let's check if our convolve function works correctly
print(training_data.shape)
image = training_data[0:10,:,:]
print(image.shape)
convolved_features = Convolve(patch_size,length_hidden,image,W_1_Best,B_1_Best)
check_convolve(image,convolved_features,W_1_Best,B_1_Best)


#Next let's see if our Pooling function works correctly
pooled_features = Pool(poolDim,convolved_features)
check_pool()

#Fantastic they both work! Thus we can comment this section out and continue coding!!!
'''


#Now let's do our convolution and pooling!

for i in range(length_hidden/STEP_SIZE):   #Because we are iterating for 25 features at a time (so the code isn't incredibly slow and so it doesn't run out of memory) across 100 features we loop back 4 different times

	#First let's go ahead and pull out our theta weights that are appropriate to how far along we are in iterating across these features
	W1_temp = W_1_Best[i*STEP_SIZE:i*STEP_SIZE + STEP_SIZE,:] #DIMENSIONS OF (25,225)
	
	B1_temp = B_1_Best[i*STEP_SIZE:i*STEP_SIZE + STEP_SIZE,:] #DIMENSIONS OF (25,1)

	#Now that we have grabbed out weight and bias terms for this section of convolution and pooling, lets actually convolve and pool the data!
	
	#First we convolve the features 
	#Convolve has inputs of (patchDim, numFeatures, images, W, B)

	convolvedFeatures_train = Convolve(patch_size,STEP_SIZE,training_data,W1_temp,B1_temp)

	#Next we pool the convolvedFeatures
	#Pool has inputs of (poolDim,convolvedFeatures)
	PoolFeatures_train = Pool(poolDim,convolvedFeatures_train)
	
	#Next we assign these pooled convolved features (mean activation over the 3x3 regions of convolved features) to our matrix of pooled convolved features for the entire training set
	pooling_training[i*STEP_SIZE:i*STEP_SIZE + STEP_SIZE,:,:,:] = PoolFeatures_train

	#We must also repeat this process for the test set as well
	
	convolvedFeatures_test = Convolve(patch_size,STEP_SIZE,testing_data,W1_temp,B1_temp)

	PoolFeatures_test = Pool(poolDim,convolvedFeatures_test)
	
	pooling_testing[i*STEP_SIZE:i*STEP_SIZE + STEP_SIZE,:,:,:] = PoolFeatures_test
	

#Now let's output these pooled features to a different data file so that we can use it later for classification
pooled_training_list = np.ravel(pooling_training)
print(pooled_training_list.shape)
pooled_testing_list = np.ravel(pooling_testing)
print(pooled_testing_list.shape)
np.savetxt('output_folder/MNISTTrain_Set_CONVED_AND_POOLED_STEP_25_PoolDim_7' + str(l) + '_B_' + str(Beta) + '_Rho_' + str(P) + '.out', pooled_training_list)

np.savetxt('output_folder/MNISTTest_Set_CONVED_AND_POOLED_STEP_25_PoolDim_7' + str(l) + '_B_' + str(Beta) + '_Rho_' + str(P) + '.out', pooled_testing_list)

############### THINGS TO DO ###############
#Test many parameters
