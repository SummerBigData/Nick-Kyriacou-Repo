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


def Convolve(patchDim, numFeatures, images, W, B):
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
	num_images = len(images)

	# Dimension of image (28)
	img_dim = len(images[0,:])

	num_combos = 1 + img_dim - patchDim #Number of possible samples we can look at for each 28 feature image, (14)

	convolved_features = np.zeros((numFeatures, num_images, num_combos, num_combos)) # This is our matrix of convovled features(25, num_images, 14, 14)

	# First lets initializes the matrices used during the convolution to be full of zeros and we can fill them in later
	# We need to compute WT, where W is our weights and T is our whitening matrix, we also need to  compute B - WTx_bar
	# B is the bias term and x_bar is the mean patch

	
	W_ordered = np.reshape(W,(len(W), patchDim, patchDim)) #DIMENSIONS OF: (100,15,15)

	
	for i in range(num_images): #Loops through every image
		
		for j in range(numFeatures): #Loops through every feature within the image
			# Let's create a pre-matrix will zeros for each element
			# This matrix corresponds to the feature matrix for each image that we will fill one for each channel
			
			convolved_image = np.zeros((num_combos, num_combos))   # DIMENSIONS OF: (14, 14)
			
			
			# We must next grab the feature needed during the convolution (patch_dim, patch_dim)  (15,15)
			# First initialize an empty matrix of zeros for this we can fill later with the ordered WT values
			feature = np.zeros((patchDim, patchDim))  # DIMENSIONS OF: (15, 15)
			
			feature = W_ordered[j, :, :] #Takes the given feature for a given image
			
			# Flip the feature matrix because convolutions mathematical definition involves flipping the matrix to convolve with
			feature = np.flipud(np.fliplr(np.squeeze(feature)))
	
			# Obtain the image that we will convolve
			temp_image = np.squeeze(images[i,:,:])         # DIMENSIONS OF: (64, 64)
				
			# Convolve feature with temp_image, add the result to convolved_image
			convolved_image += scipy.signal.convolve2d(temp_image, feature, mode = 'valid', boundary = 'fill', fillvalue = 0)	# DIMENIONS OF: (57, 57)


			
			convolved_image = convolved_image + B[j, 0]    # We don't convolve with the bias term so now we must add this term back onto it 

			# Apply the sigmoid function to get the hidden activation
			# The convolved feature is the sum of the convolved values for all channels
			convolved_features[j, i, :, :] = sigmoid(convolved_image)

	return convolved_features
