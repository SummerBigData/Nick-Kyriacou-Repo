#This code generates the training set. It does this by randomly picking one of the 1 images.
#Next it samples randomly an 8x8 image patch from the selected image, and converts the patch into a 64-dimensional vector to get a single training example
#It will do this for 10000 image patches and concatenate these training examples into a 64x10000  matrix. A good portion of this code is shamelessly stolen from suren's repository.
#Owner: Nick Kyriacou
#Created: 6/12/2018

#IMPORTING PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
from random import randint

name = 'output_folder/10000Random8x8.out'

#FUNCTION DEFINITIONS

# Plot the entire 10 image dataset, 5 images in a row, two in a column
def PlotAll(dat):
	pic1 = dat[0]
	for i in range(4):
		pic1 = np.concatenate((pic1, dat[i+1]), axis = 1)

	pic2 = dat[5]
	for i in range(4):
		pic2 = np.concatenate((pic2, dat[i+4]), axis = 1)

	PlotImg(np.vstack((pic1, pic2)) )
#	imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
#	plt.show()

def PlotImg(mat):
	imgplot = plt.imshow(mat, cmap="binary", interpolation='none') 
	plt.show()








def savePics(vec): #This saves the Pictures into an output data file
	np.savetxt(name, vec, delimiter=',')


#####################################MAIN CODE STARTS HERE#####################################################3

#First step is to load data from matlab file

data_original = scipy.io.loadmat('IMAGES.mat')#Loads the data
patch_size = 8 #We want 8x8 image patch
total_images = 10000 #We want to sample 10000 image patches





# To see how long the code runs for, we start a timestamp
print 'Running randpicGen.py'
#print 'Will be saved in: ', saveStr


# The data is originally in a 512 x 512 x 10, so we must convert it to 10 x 512 x 512. This corresponds to 10 images total, 512x512 pixels each

data_temp = np.zeros((10, 512, 512))
for i in range(512):
	for j in range(512):
		for k in range(10):
			data_temp[k,i,j] = data_original['IMAGES'][i,j,k]
	
# For plotting the data
#PlotAll(data_temp)

sampleDat = np.zeros((total_images, 8, 8))
for i in range(total_images):
	wPic = randint(0, 9) # Pick one of the 10 images
	wRow = randint(0,504)# Pick a row for the first element
	wCol = randint(0,504)# Pick a column for the first element

#		# For plotting one of the images
	imgplot = plt.imshow(data_temp[wPic,wRow:wRow+8,wCol:wCol+8], cmap="binary", interpolation='none') 
	plt.show()
	sampleDat[i]=data_temp[wPic,wRow:wRow+8,wCol:wCol+8]
	

savePics(np.ravel(sampleDat))





'''
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

image2 = np.reshape(training_set[:,0:1], (8, 8))
plt.imshow(image2, cmap = 'binary', interpolation = 'none')
plt.show()

#print training_set.shape

# Save our patches array to a file to be used later
training_set = np.ravel(training_set)
name = 'output_folder/' + str(total_images) +'Random8x8.out'
np.savetxt(name, training_set, delimiter = ',')
'''
