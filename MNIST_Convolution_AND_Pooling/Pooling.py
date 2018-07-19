#Purpose: This code contains the Pool function and performs Pooling when called
#Created By: Nick Kyriacou
#Created on: 6/25/2018



######################## IMPORTING PACKAGES #################################
import numpy as np
import scipy.signal


########################## FUNCTION DEFINITIONS START HERE #########################


def Pool(poolDim,convolvedFeatures):
	
	'''
	Parameter information borrowed from sample code.
	Input Parameters:
	poolDim = dimension of pooling region  = (19)
	convolvedFeatures = convolved features to pool. This has dimensions of (400,m,57,57) = (length_hidden,num_images,num_combos,num_combos)
	Output Parameters:
	This function returns a matric of pooled features (pooledFeatures) with dimensions (400, m, 3, 3)
	'''
	
	#First let's decide the size  of the region to pool our convolved features over

	num_images = len(convolvedFeatures[0,:,0,0])

	num_features = len(convolvedFeatures[:,0,0,0])

	convolvedDim = len(convolvedFeatures[0,0,:,0])
	#Then we divide our convolved features into disjoint m x n regions
	#We decided that we can divide our convolved features into 9, disjoint, 3x3 regions. (This was chosen because 3 is the only LCM of 57)
	
	size_pooling_regions = int(np.floor(convolvedDim/poolDim)) # (3) Is the length of our pooling regions. There is a 3x3 grid of pooling regions.

	pooledFeatures = np.zeros((num_features,num_images,size_pooling_regions,size_pooling_regions)) # Initializes matrix of zeros for pooledFeatures of dimensions (400,num_images,3,3) 
	#pooledFeatures is what the function will return
	
 	#Next we take the mean feature activation over these regions to obtain the pooled convolved features. We use these features for classification

	for i in range(size_pooling_regions): #Let's think of this as an interation over each columns
	
		for k in range(size_pooling_regions): #Let's think of this as interating over each row	
			temp_poolFeatures_mat = np.zeros((num_features,num_images,size_pooling_regions,size_pooling_regions))
			temp_poolFeatures_mat = convolvedFeatures[:,:,i*poolDim:i*poolDim + poolDim,k*poolDim:k*poolDim + poolDim]
			pooledFeatures[:,:,i,k] = np.mean(temp_poolFeatures_mat,axis = (2,3)) #Finds the mean feature activation about this axis of the pooledFeatures

	return(pooledFeatures)
