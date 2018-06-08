#This code generates the training set. It does this by randomly picking one of the 1 images.
#Next it samples randomly an 8x8 image patch from the selected image, and converts the patch into a 64-dimensional vector to get a single training example
#It will do this for 10000 image patches and concatenate these training examples into a 64x10000  matrix
#Owner: Nick Kyriacou
#Created: 6/7/2018

#IMPORTING PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
import time



#FUNCTION DEFINITIONS




#MAIN CODE STARTS HERE

#First step is to load data from matlab file

data = scipy.io.loadmat('IMAGES.mat')#Loads the data
images = data['IMAGES']
patch_size = 8 #We want 8x8 image patch
total_images = 10000 #We want to sample 10000 image patches

#Now let's create a 64x10000 matrix that we want
training_set = np.zeros((patch_size**2,total_images)) 
for i in range(len(training_set[0])):
	int_random = random.randint(0, 504)
	get_random_image_num = random.randint(0, 9) #randomly chooses an image
	temp_image = images[:,:, get_random_image_num] #randomly picks an image from the set of 10
	temp = temp_image[int_random: int_random + 8, int_random: int_random + 8]
	temp = np.reshape(temp, (64, 1))
	training_set[:, i:i+1] = temp


# Prints out an 8x8 pixellated image

#image2 = np.reshape(training_set[:,0:1], (8, 8))
#plt.imshow(image2, cmap = 'binary', interpolation = 'none')
#plt.show()

#print training_set.shape

# Save our patches array to a file to be used later
training_set = np.ravel(training_set)
name = 'output_folder/' + str(total_images) +'Random8x8.out'
np.savetxt(name, training_set, delimiter = ',')
