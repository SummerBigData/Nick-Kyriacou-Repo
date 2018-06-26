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
from Create_test_and_train_images import generate_images
from Convolution import Convolve
from Pooling import Pool
import random
import sys

################################## GLOBAL CONSTANT DEFINITIONS #######################
features = 192
length_hidden = 400
patch_size = 8 # 8x8
num_colors = 3 # (RGB)
imageDim = 64 #64x64 pixel images
poolDim = 19 #Dimension of Pooling region
STEP_SIZE = 25 # Convolve and Pool only 50 features at a time to not run out of memory
########################### FUNCTION DEFINITIONS #############################

def seperate(theta_all):
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features)) # (400,192)
	B_1 = np.reshape(theta_all[length_hidden*features:features*length_hidden + length_hidden],(length_hidden,1)) # (400,1)

	W_2 = np.reshape(theta_all[features*length_hidden + length_hidden: 2*features*length_hidden + length_hidden],(features,length_hidden)) # (192,400)
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1)) # (192,1)

	return(W_1,B_1,W_2,B_2)


def sigmoid(z): 
	hypothesis = 1.0/(1.0 + np.exp(-z) )
	return(hypothesis)

def convolution_time():
	#First we want to compute W*x(r,c) + B  for all possible (8x8) patch sizes
	#To do this step we have a scipy function scipy.signal.convolve2d which allows us to optimally compute W*x(r,c) + B for all patch sizes 


	
	#scipy.signal.convolve2d(input_1,input_2,mode,boundary,fill-value) #5 different arguments

	return()



def check_convolve(image,convolved_feature,W_1,B_1,ZCA,mean): #This code checks whether my convolution function did what it was supposed to (form borrowed from Matt)
	num_images = len(image[0,0,0,:])

	for i in range(1000):
		# Pick a random data patch
		feature_num = np.random.randint(0, length_hidden - 1)
		image_num = np.random.randint(0, 9)
		image_row = np.random.randint(0, imageDim - patch_size)
		image_col = np.random.randint(0, imageDim - patch_size)
	
		patch = image[image_row: image_row + patch_size, image_col: image_col + patch_size, : ,image_num]
		# Flatten the image into seperate groups of color
		patch = np.concatenate((patch[:,:,0].flatten(), patch[:,:,1].flatten(), patch[:,:,2].flatten()))
				
		patch -= np.ravel(mean)
		patch = patch.reshape(1,192)
		patch = np.dot(patch, zca_white_mat)
		
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


#First let's get all our data that was previously generated
best_weights = np.genfromtxt('data/optimal_thetas_l_0.003_Beta_5.0_Rho_0.035_.out',dtype = float) 
mean = np.genfromtxt('data/stlSampledPatches_mean',dtype=float)
zca_white_mat = np.genfromtxt('data/stlSampledPatches_ZCA_White_Matrix',dtype = float)
training_images,training_labels,test_images,test_labels = generate_images()
num_training_samples = len(training_images[0,0,0,:])
num_test_samples = len(test_images[0,0,0,:])

#Let's reshape these vector lists back into matrices
mean = np.reshape(mean,(len(mean),1)) # (192,1) Matrix
zca_white_mat = np.reshape(zca_white_mat,(features,features)) # (192,192) Matrix 

#Next let's unpackage our theta weights
W_1_Best,B_1_Best,W_2_Best,B_2_Best = seperate(best_weights)

print(W_1_Best.shape)
print(B_1_Best.shape)
print(W_2_Best.shape)
print(B_2_Best.shape)

print('shapes')
print(mean.shape)
print(zca_white_mat.shape)

print(training_images.shape)
print(test_images.shape)
'''
pooled_features_train = np.zeros((global_hidden_size, num_train_images,
	np.floor((global_image_dim - global_patch_dim + 1) / global_pool_dim),
	np.floor((global_image_dim - global_patch_dim + 1) / global_pool_dim)))

pooled_features_test = np.zeros((global_hidden_size, num_test_images,
	np.floor((global_image_dim - global_patch_dim + 1) / global_pool_dim),
np.floor((global_image_dim - global_patch_dim + 1) / global_pool_dim)))
'''
#Now that we have loaded in our data let us next take a look at creating our matrix we will use to pool Dimensions of (length_hidden,num_samples,num_pooling_regions,num_pooling_regions)
pooling_training = np.zeros((length_hidden,num_training_samples,(64 - 8 + 1)/poolDim,(64 - 8 + 1)/poolDim)) # (400,2000,3,3)
pooling_testing = np.zeros((length_hidden,num_test_samples,(64 - 8 + 1)/poolDim,(64 - 8 + 1)/poolDim))      # (400,3200,3,3)



'''
#First let's check if our convolve function works correctly
image = training_images[:,:,:,0:10]
print(image.shape)
convolved_features = Convolve(patch_size,length_hidden,image,W_1_Best,B_1_Best,zca_white_mat,mean)
check_convolve(image,convolved_features,W_1_Best,B_1_Best,zca_white_mat,mean)


#Next let's see if our Pooling function works correctly
pooled_features = Pool(poolDim,convolved_features)
check_pool()

#Fantastic they both work! Thus we can comment this section out and continue coding!!!
'''








#Now let's do our convolution and pooling!

for i in range(length_hidden/STEP_SIZE):   #Because we are iterating for 25 features at a time (so the code isn't incredibly slow and so it doesn't run out of memory) across 400 features we loop back 8 different times

	#First let's go ahead and pull out our theta weights that are appropriate to how far along we are in iterating across these features
	W1_temp = W_1_Best[i*STEP_SIZE:i*STEP_SIZE + STEP_SIZE,:] #DIMENSIONS OF (25,192)
	B1_temp = B_1_Best[i*STEP_SIZE:i*STEP_SIZE + STEP_SIZE,:] #DIMENSIONS OF (25,1)

	#Now that we have grabbed out weight and bias terms for this section of convolution and pooling, lets actually convolve and pool the data!
	
	#First we convolve the features 
	#Convolve has inputs of (patchDim, numFeatures, images, W, B, ZCAWhite, meanPatch)

	convolvedFeatures_train = Convolve(patch_size,STEP_SIZE,training_images,W1_temp,B1_temp,zca_white_mat,mean)

	#Next we pool the convolvedFeatures
	#Pool has inputs of (poolDim,convolvedFeatures)
	PoolFeatures_train = Pool(poolDim,convolvedFeatures_train)
	
	#Next we assign these pooled convolved features (mean activation over the 3x3 regions of convolved features) to our matrix of pooled convolved features for the entire training set
	pooling_training[i*STEP_SIZE:i*STEP_SIZE + STEP_SIZE,:,:,:] = PoolFeatures_train

	#We must also repeat this process for the test set as well
	convolvedFeatures_test = Convolve(patch_size,STEP_SIZE,test_images,W1_temp,B1_temp,zca_white_mat,mean)

	PoolFeatures_test = Pool(poolDim,convolvedFeatures_test)
	
	pooling_testing[i*STEP_SIZE:i*STEP_SIZE + STEP_SIZE,:,:,:] = PoolFeatures_test


#Now let's output these pooled features to a different data file so that we can use it later for classification
pooled_training_list = np.ravel(pooling_training)
pooled_testing_list = np.ravel(pooling_testing)
np.savetxt('output_folder/Train_Set_CONVED_AND_POOLED_STEP_25.out', pooled_training_list)
np.savetxt('output_folder/Test_Set_CONVED_AND_POOLED_STEP_25.out', pooled_testing_list)
############### THINGS TO DO ###############
'''
1) Make convolution function
2) Do everything else :P
3)Implement pooling
4) check convolution
5) implement check pooling function
'''
