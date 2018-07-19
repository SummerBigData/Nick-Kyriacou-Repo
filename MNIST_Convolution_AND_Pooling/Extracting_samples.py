#Purpose: This code will extract 10,000 15x15 randomly sampled patches of images from the MNIST data-set
#Created by: Nick Kyriacou
#Created on: 6/27/2018



###################### IMPORTING PACKAGES ##########################

import numpy as np
import scipy.io 
import gzip
import struct as st
import random
from random import randint

################# FUNCTION DEFINITIONS START HERE ###########################





def read_ids(filename, n = None): #This function used to read MNIST dataset
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB',f.read(4))
		shape  = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(),dtype = np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
			return arr

################# MAIN CODE STARTS HERE ##################

rand_image_dim = 15
MNIST_image_dim = 28

training_data = read_ids('data/train-images-idx3-ubyte.gz',60000)
input_size = len(training_data)

#Now that we have the data let's seed the random number generator
training_data = training_data/255.0 #This normalizes the data which is important for later 
np.random.seed(10)

training_data = np.reshape(training_data,(input_size,MNIST_image_dim,MNIST_image_dim)) # Training data has dimensions of (60000,28,28)



output_data_samples = 10000

#Next let's extract a random 15x15 patch of pixels from one image
random_samples = np.zeros((output_data_samples,rand_image_dim,rand_image_dim)) #Creates matrix of zeros with Dim of (10000,15,15)


for i in range(output_data_samples):
	#This will randomly select an image to sample
	sampled_image = random.randint(0,input_size - 1) #Finds a random integer from 0 to 59999 corresponding to which image to sample from the training data
 	
	selected_random_image = training_data[sampled_image,:] #This gives us our random image
	#print(selected_random_image.shape)
	
	#Now lets generate a random number to determine which patches to sample from, we restrict it to 13 size so that we don't overextend sampling into a feature that doesn't exist
	rand_patch_size = random.randint(0,MNIST_image_dim - rand_image_dim)
	random_patch = selected_random_image[rand_patch_size:rand_patch_size  +rand_image_dim,rand_patch_size: rand_patch_size+rand_image_dim]

	random_samples[i,:] = random_patch


#Now let's reshape this and package it out to be used later
random_samples = np.reshape(random_samples,(output_data_samples,rand_image_dim**2))

random_samples = np.ravel(random_samples)
output_name = 'output_folder/randomly_sampled_10k_15x15_pixel_images.out'
#Finally write this out to a file
np.savetxt(output_name,random_samples,delimiter = ',')
