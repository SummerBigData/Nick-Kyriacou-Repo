#Purpose: This code contains the Convolve function and performs Convolution when called
#Created By: Nick Kyriacou
#Created on: 6/25/2018

######################## IMPORTING PACKAGES #################################
import numpy as np
import scipy.signal


########################## FUNCTION DEFINITIONS START HERE #########################
def sigmoid(value):
	hypothesis = 1.0/(1.0+np.exp(-1.0*value))
	return(hypothesis)


def Convolve(patchDim, numFeatures, images, W, B, ZCAWhite, meanPatch):
	'''
	Discussion of Codes Input Parameters:
	patchDim = dimension (8) of patch region (8x8)
	numFeatures = amount of features we are using (400) =  size of hidden layer
	images = matrix of our images with Dimensions( 64, 64, 3,m)
	W = The matrix of weights (400, 192)
	B =  The Bias term (400, 1)
	ZCAWhite = The whitening matrix we computed earlier (192, 192)
	meanPatch = the mean of our patches that we computed earlier (1, 192)
	
	This function returns
	convolved_features = matrix of convolved features with dimensions (400, m, 57, 57)
	'''
	# Number of images
	num_images = len(images[0,0,0,:])
	# Dimension of image
	img_dim = len(images[0,:,0,0])
	# Number of channels, Each different channel corresponds to either RGB
	num_channel = len(images[0,0,:,0])
	num_combos = 1 + img_dim - patchDim #Number of possible samples we can look at for each 64 feature image, (57)
	convolved_features = np.zeros((numFeatures, num_images, num_combos, num_combos)) # This is our matrix of convovled features(400, num_images, 57, 57)

	# First lets initializes the matrices used during the convolution to be full of zeros and we can fill them in later
	# We need to compute WT, where W is our weights and T is our whitening matrix, we also need to  compute B - WTx_bar
	# B is the bias term and x_bar is the mean patch

	WT = np.dot(W, ZCAWhite.T)                  # DIMENSIONS OF: (400, 192)
	Bias_mean = B - np.dot(WT, meanPatch)         # DIMENSIONS OF: (400, 1)
	# Next we want to reform WT into a different size matrix to make calculations easier (400,8,8,3)
	s = patchDim ** 2  # (s = 64)
	WT_ordered = np.zeros((len(WT), patchDim, patchDim, num_channel )) #DIMENSIONS OF: (400,8,8,3)
	
	for i in range(len(WT)):
		#This just ensures that the number of channels is correctly ordered 
    		WT_ordered[i,:,:,0] = WT[i, 0:64].reshape(patchDim, patchDim)
   		WT_ordered[i,:,:,1] = WT[i, 64:128].reshape(patchDim, patchDim)
    		WT_ordered[i,:,:,2] = WT[i, 128:].reshape(patchDim, patchDim)

	
	for i in range(num_images): #Loops through every image
		
		for j in range(numFeatures): #Loops through every feature within the image
			# Let's create a pre-matrix will zeros for each element
			# This matrix corresponds to the feature matrix for each image that we will fill one for each channel
			
			convolved_image = np.zeros((num_combos, num_combos))   # DIMENSIONS OF: (57, 57)
			
			
			for k in range(num_channel): #Loops through every color for each feature 
				# We must next grab the feature needed during the convolution (patch_dim, patch_dim)  (8,8)
				# First initialize an empty matrix of zeros for this we can fill later with the ordered WT values
				feature = np.zeros((patchDim, patchDim))  # DIMENSIONS OF: (8, 8)
				feature = WT_ordered[j, :, :, k] #Takes the given feature and color value for a given image
	
				# Flip the feature matrix because convolutions mathematical definition involves flipping the matrix to convolve with
				feature = np.flipud(np.fliplr(np.squeeze(feature)))
	
				# Obtain the image that we will convolve
				temp_image = np.squeeze(images[:, :, k,i])         # DIMENSIONS OF: (64, 64)
				
				# Convolve feature with temp_image, add the result to convolved_image
				convolved_image += scipy.signal.convolve2d(temp_image, feature, mode = 'valid', boundary = 'fill', fillvalue = 0)	# DIMENIONS OF: (57, 57)


			
			convolved_image += Bias_mean[j, 0]    # We don't convolve with the bias term so now we must add this term back onto it 

			# Apply the sigmoid function to get the hidden activation
			# The convolved feature is the sum of the convolved values for all channels
			convolved_features[j, i, :, :] = sigmoid(convolved_image)

	return convolved_features
