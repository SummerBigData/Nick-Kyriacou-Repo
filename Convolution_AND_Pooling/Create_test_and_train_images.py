#Purpose: The purpose of this script is to generate and show some of the test and training images that will be used for the convolution and pooling exercise.
#Created By: Nick Kyriacou
#Created on: 6/22/2018



################################# IMPORTING PACKAGES ########################


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io
from scipy.optimize import minimize 
import scipy.signal

################## VARIABLE DEFINITIONS ######################
patch_size = 8
num_colors = 3

################### Function Definition #########################

def generate_images():

	#First we want to load the testing data
	test_data = scipy.io.loadmat('data/stlTestSubset.mat')
	test_images = test_data['testImages'] # (64,64,3,3200)
	test_labels = test_data['testLabels'] # (3200,1)
	
	num_test_images = 3200
	
	#Next let's generate the training data 
	train_data = scipy.io.loadmat('data/stlTrainSubset.mat')
	train_images = train_data['trainImages'] # (64,64,3,2000)
	train_labels = train_data['trainLabels'] # (2000,1)
	
	num_train_images = 2000	
	
	#Let's take a look at some of the training and test images for fun!

	train_row_1 = np.concatenate((train_images[:,:,:,20],train_images[:,:,:,25],train_images[:,:,:,30],train_images[:,:,:,35],train_images[:,:,:,40]),axis = 1)
	train_row_2 = np.concatenate((train_images[:,:,:,45],train_images[:,:,:,50],train_images[:,:,:,55],train_images[:,:,:,60],train_images[:,:,:,65]),axis = 1)
	train_row_3 = np.concatenate((train_images[:,:,:,70],train_images[:,:,:,85],train_images[:,:,:,90],train_images[:,:,:,95],train_images[:,:,:,100]),axis = 1)
	train_row_4 = np.concatenate((train_images[:,:,:,105],train_images[:,:,:,110],train_images[:,:,:,115],train_images[:,:,:,120],train_images[:,:,:,125]),axis = 1)
	train_row_5 = np.concatenate((train_images[:,:,:,130],train_images[:,:,:,135],train_images[:,:,:,140],train_images[:,:,:,145],train_images[:,:,:,150]),axis = 1)
	train_row_6 = np.concatenate((train_images[:,:,:,155],train_images[:,:,:,160],train_images[:,:,:,165],train_images[:,:,:,170],train_images[:,:,:,175]),axis = 1)

	train_panel = np.concatenate((train_row_1,train_row_2,train_row_3,train_row_4,train_row_5,train_row_6),axis = 0)


	test_row_1 = np.concatenate((test_images[:,:,:,20],test_images[:,:,:,25],test_images[:,:,:,30],test_images[:,:,:,35],test_images[:,:,:,40]),axis = 1)
	test_row_2 = np.concatenate((test_images[:,:,:,45],test_images[:,:,:,50],test_images[:,:,:,55],test_images[:,:,:,60],test_images[:,:,:,65]),axis = 1)
	test_row_3 = np.concatenate((test_images[:,:,:,70],test_images[:,:,:,85],test_images[:,:,:,90],test_images[:,:,:,95],test_images[:,:,:,100]),axis = 1)
	test_row_4 = np.concatenate((test_images[:,:,:,105],test_images[:,:,:,110],test_images[:,:,:,115],test_images[:,:,:,120],test_images[:,:,:,125]),axis = 1)
	test_row_5 = np.concatenate((test_images[:,:,:,130],test_images[:,:,:,135],test_images[:,:,:,140],test_images[:,:,:,145],test_images[:,:,:,150]),axis = 1)
	test_row_6 = np.concatenate((test_images[:,:,:,155],test_images[:,:,:,160],test_images[:,:,:,165],test_images[:,:,:,170],test_images[:,:,:,175]),axis = 1)
	
 	test_panel = np.concatenate((test_row_1,test_row_2,test_row_3,test_row_4,test_row_5,test_row_6),axis = 0)	

	print('first showing 30 training_images')
	plt.imshow(train_panel, interpolation = 'none')
	plt.show()
	print('next are 30 of the test images')
	plt.imshow(test_panel,interpolation= 'none')
	plt.show()
	


	return(train_images,train_labels,test_images,test_labels)




