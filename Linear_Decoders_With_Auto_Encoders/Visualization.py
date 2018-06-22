#PURPOSE: This code takes the thetas values optimized by the Sparse Autoencoder with a Linear Decoder hypothesis function and visualizes the learned features. The visualization should look like edges and "opponent colors".
#CREATED ON: 6/19/2018
#CREATED BY: Nick Kyriacou


###################### IMPORTING PACKAGES #######################

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.image as mpimg
from scipy.optimize import minimize 
import struct as st
import gzip

########################## FUNCTION DEFINITIONS ########################

def seperate(theta_all): #This function seperates the total theta vector into its individual components
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features))
	B_1 = np.reshape(theta_all[features*length_hidden:features*length_hidden + length_hidden],(length_hidden,1))
	W_2 = np.reshape(theta_all[features*length_hidden+length_hidden:2*features*length_hidden+length_hidden],(features,length_hidden))
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1))
	
	return(W_1,B_1,W_2,B_2)


################## GLOBAL VARIABLE INITIALIZATIONS AND IMPORTING DATA ###########################

training_sets = 100000
features = 192
length_hidden = 400
l = 0.003 #Lambda Parameter #CHANGE TO READ IN NEW DATA
Beta  = 5.0 #Beta Parameter #CHANGE TO READ IN NEW DATA
P = 0.035 #Rho Parameter #CHANGE TO READ IN NEW DATA
output_name = 'output_folder/optimal_thetas_l_' + str(l)+ '_Beta_' + str(Beta) + '_Rho_' + str(P) + '_.out'

best_thetas = np.genfromtxt(output_name,dtype = float)
raw_input_data = scipy.io.loadmat('data/stlSampledPatches.mat')
raw_input_data = raw_input_data['patches'].T
input_images_whitened = np.genfromtxt('data/stlSampledPatches_ZCA_Whitened.out',dtype = float)

W_1_best,B_1_best,W_2_best,B_2_best = seperate(best_thetas)
######################### MAIN CODE STARTS HERE ######################


################### VISUALIZING THE MAXIMUM ACTIVATIONS ################
ZCA_Whitening_mat = np.genfromtxt('data/stlSampledPatches_ZCA_White_Matrix',dtype=float)
ZCA_Whitening_mat = np.reshape(ZCA_Whitening_mat,(features,features)) # (192x192)

#Now we calculate the activation units
numerator = W_1_best
print(numerator.shape)
denominator = np.sqrt(np.sum(W_1_best**2,axis = 1))
print denominator.shape
activation = numerator/np.reshape(denominator,(len(W_1_best),1))
print(activation)
activation = np.dot(activation,ZCA_Whitening_mat)
activation = (activation + 1.0)/2.0 #This functionally renormalizes the activation image

#Next lets start creating our panel of 20x20 images to visualize the learned color features
columns = 20
rows = 20
black_space_horizontal = np.ones((2, 8*rows+2*(columns-1), 3))*np.amax(activation)
black_space_vertical = np.ones((8, 2, 3))*np.amax(activation)
panel = black_space_horizontal #Every time we concatenate a different panel we should put some horizontal blackspace between each image

#One tricky thing to note is how these RBG images are stored. In each 192 length vector the first 64 digits correspond to , the second 64 digits correspond to and the final 64 digits correspond to . Thus we should break up our individual images into an 8x8x3 matrix.




#We will stitch together 20 sets of images for each row and then concatenate them by column
for i in range(rows): 
	#Each different row will Initially start out with a vertical border of blackspace
	for k in range(columns):
			# First set up our image in red, then blue, then green
		
    		individual_image = np.zeros((8, 8, 3))
    		individual_image[:,:,0] = activation[k + i * rows][0:64].reshape(8,8)
   		individual_image[:,:,1] = activation[k + i * rows][64:128].reshape(8, 8)
    		individual_image[:,:,2] = activation[k + i * rows][128:].reshape(8, 8)
			
		if (k == 0): #This code recognizes that there is nothing to stack the first time and thus creates a placeholder image to stack for future iterations
			row_panel = individual_image #This first assignment acts as a placeholder
		else: 
			row_panel = np.concatenate((row_panel, black_space_vertical, individual_image), axis = 1) #Let's stack together the images together by column for each of the twenty columns
			
	if (i == 0):
		images2 = row_panel #This is a placeholder because for first iteration we have nothing to stack onto
	else:
		images2 = np.concatenate((images2, black_space_horizontal, row_panel), axis = 0) #Let's stack together our twenty columns images every time we go on to a new row

#Next let's display this grid of 20x20 images
imgplot = plt.imshow(images2,interpolation='nearest') 
plt.savefig('Picture_folder/activations_for_node_hidden_layer_Lamb_'+str(l)+'_Beta_'+str(Beta) +'_Rho_'+str(P) +'.png',transparent=False, format='png')
plt.show()






















'''
		individual_image = np.zeros((8,8,3)) #corresponds to 8 pixels by 8 pixels by RBG color
		#Now let's determine which values of the images are Red, Blue,and Green and store them in the appropriate index of the matrix
		individual_image[:,:,0] = activation[k + i*rows][0:64].reshape(8,8) # Because each unraveled RBG list has 64 components we reshape this into a RBG array
		individual_image[:,:,1] = activation[k + i*rows][64:128].reshape(8,8)
		individual_image[:,:,2] = activation[k + i*rows][128:].reshape(8,8)
		
		#print(row_pic.shape)
		#print(black_space_vertical.shape)
		#print(individual_image[i*rows+k].shape)
		row_pic = np.concatenate((row_pic,individual_image,black_space_vertical),axis = 1)
	print('shapes')
	print(panel.shape)
	print(row_pic.shape)
	print(black_space_horizontal.shape)
	panel = np.concatenate((panel,row_pic,black_space_horizontal),axis = 0)
'''
		#Now we want to differentiate between whether or not this is the first panel we are creating
#GENERAL PROCESS
'''
1) FEED FORWARD USING OPTIMAL WEIGHTS TO GET OUTPUT UNITS AND HIDDEN LAYER ACTIVATION UNITS
2) NORMALIZE INPUTS AND OUTPUTS USING NORMALIZE FUNCTION
3) (OPTIONAL) CALCULATE AVG ERROR = AVG DEVIATION FROM INPUT TO OUTPUT LAYER (FOR SPARSE AUTOENCODER INPUT SHOULD EQUAL OUTPUTS)
4) (OPTIONAL) PLOT IMAGE OF INPUT AND OUTPUT LAYER SIDE BY SIDE TO SEE HOW ACCURATE WE ARE VISUALLY
5) VISUALIZE THE MAX ACTIVATIONS
6) THIS IS DONE BY FIRST PULLING OUT AND RESHAPING THE WHITENING MATRIX TO A (FEATURES,FEATURES) (192,192)
7) NOW WE FIND THE MAXIMUM ACTIVATIONS IN THE SAME WAY THAT WE DID IN THE PREVIOUS EXERCISE
8) NEXT WE MUST CREATE 400 IMAGES (20 ROWS X 20 COLUMNS)
9)
'''
