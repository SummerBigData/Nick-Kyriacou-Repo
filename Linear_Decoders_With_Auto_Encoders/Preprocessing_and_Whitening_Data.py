#PURPOSE: The purpose of this script will be to pre-process and whiten our input data so that we can effectively use a Sparse Auto Encoder onto it. It will take the original data and store the processed data in an output folder. Additionally this script will make sure each value in the data is between 0 and 1 (Normalizing it, because these are the only values accepted by our hypothesis function)
#CREATED ON: 6/19/2018
#CREATED BY: Nick Kyriacou

################################### IMPORTING PACKAGES ################################


import numpy as np
import time
import scipy.io

def ZCA_Whitening(data): #This function whitens the data through a complicated process that I do not understand. Code is borrowed from Matt's Repo.
	print('whitening')
	# The first step is to subtract the mean from the input matrix
	mean = np.mean(data, axis = 0).reshape(1, data.shape[1])  # DIMENSIONS OF (1, 192) = (1,features)
	print(mean.shape)
	data -= np.tile(mean, (data.shape[0], 1)) 	            # DIMENSIONS OF (10000, 192) = (sample_size, features)
	print(data.shape)

	# Next we should whiten our input matrix
	sigma = np.dot(data.T, data) / float(len(data))
	u, s, v = np.linalg.svd(sigma)
	ZCAWhite = np.dot(np.dot(u, np.diag(1.0 / np.sqrt(s + epsilon))), u.T)
	print(ZCAWhite.shape)
	whitened_patches = np.dot(data, ZCAWhite)
	print(whitened_patches.shape)
	return whitened_patches, ZCAWhite, mean # whitened_patches is our processed input, ZCAWhite and mean are two parameters that will be used in a future exercise so we are storing these for future use.


	


############################ MAIN CODE STARTS HERE ###############################
epsilon = 0.1 #This epsilon is used 
raw_data = scipy.io.loadmat('data/stlSampledPatches.mat')

#Next we should remove label ('patches') from dictionary to just get numeric values
raw_data = raw_data['patches'] #This makes data in shape of (192,10000) (features x sample_size)
print(raw_data.shape)
whitened_patches, ZCAWhite, mean = ZCA_Whitening(raw_data.T)
output_name = 'data/stlSampledPatches_ZCA_Whitened.out'
whitened_patches = np.ravel(whitened_patches)
np.savetxt(output_name,whitened_patches,delimiter = ',')
ZCAWhite = np.ravel(ZCAWhite)
mean = np.ravel(mean)
np.savetxt('data/stlSampledPatches_ZCA_White_Matrix',ZCAWhite,delimiter = ',')
np.savetxt('data/stlSampledPatches_mean',mean,delimiter = ',')

