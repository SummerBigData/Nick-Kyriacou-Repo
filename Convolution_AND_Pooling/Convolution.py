#Purpose:
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

################################## GLOBAL CONSTANT DEFINITIONS #######################
features = 192
length_hidden = 400
patch_size = 8 # 8x8
num_colors = 3 # (RGB)
data_size = 0 #size of data FIXME


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


################################ MAIN CODE STARTS HERE  ###########################


#First let's get all our data that was previously generated
best_weights = np.genfromtxt('data/optimal_thetas_l_0.003_Beta_5.0_Rho_0.035_.out',dtype = float) 
mean = np.genfromtxt('data/stlSampledPatches_mean',dtype=float)
zca_white_mat = np.genfromtxt('data/stlSampledPatches_ZCA_White_Matrix',dtype = float)

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
############### THINGS TO DO ###############
'''
1) Make convolution function
2) Do everything else :P
3) Generate Images (both training and a test set)
'''
