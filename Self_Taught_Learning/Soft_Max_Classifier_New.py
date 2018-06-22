#Purpose: The purpose of this code is to correctly implement the softmax classifier and to classify images on the test set
#Created on: 6/14/2018
#Created by: Nick Kyriacou

##################### IMPORTING PACKAGES #####################3
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
import scipy.io
import matplotlib.image as mpimg
from scipy.optimize import minimize 
import struct as st
import gzip
######################### GLOBAL VARIABLE DEFINITION #########################

gStep = 0
eps = 0.12
f1 = 784
f2 = 200
f3 = 10
rho = 0.05
beta = 0.8
lamb = 10
m = 98
#################### DATA PREP FUNCTION ###############################

def read_ids(filename, n = None): #This function used to read MNIST dataset
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB',f.read(4))
		shape  = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(),dtype = np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
			return arr


def PrepData(string): #This prepares the data to be either ranging from digits 5-9, digits 0-4 or 0-9 (the entire dataset) (Borrowed from Suren)
	# Obtain the data values and convert them from arrays to lists
		# Obtain the data values and convert them from arrays to lists
	datx = read_ids('train-images-idx3-ubyte.gz', 60000)
	daty = read_ids('train-labels-idx1-ubyte.gz', 60000)

	# Get the data in matrix form
	datx = np.ravel(datx).reshape((60000, 784))

	# Stick the data and labels together for now
	dat = np.hstack((daty.reshape(60000, 1), datx))
	
	# If the user wants the whole dataset, we can send it back now, after shuffling
	if string == '09':
		np.random.seed(5)	# Some random seed
		np.random.shuffle(dat)
		return dat[:,1:]/255.0, col(dat, 0)


	# Organize the data with respect to the labels
	ind = np.argsort( dat[:,0] ).astype(int)
	ordDat = np.zeros(dat.shape)
	for i in range(len(ind)):
		ordDat[i] = dat[ind[i]]
	#ordDat = dat[dat[:,0].argsort()]

	# Find the index of the last 4. For some reason, this is a 1 element array still, so we choose the only element [0]
	last4Ind = np.argwhere(col(ordDat, 0)==4)[-1][0]
	# Seperate the data
	dat04 = ordDat[0:last4Ind+1]
	dat59 = ordDat[last4Ind+1: ]

	# Reorder the data
	np.random.seed(7)	# Some random seed
	np.random.shuffle(dat04)
	np.random.shuffle(dat59)

	if string == '04':
		return dat04[:,1:]/255.0, col(dat04, 0)
	elif string == '59':
		return dat59[:,1:]/255.0, col(dat59, 0)
	else:
		print 'Error, input is not "04" or "59" or "09"'

def col(matrix, i): #Returns column of a certain index. Borrowed from Suren for simplicity in not having to redo data-prep code
	return np.asarray([row[i] for row in matrix])



def randMat(x, y):
	theta = np.random.rand(x,y) 	# Makes a (x) x (y) random matrix of [0,1]
	return theta*2*eps - eps # Make it range [-eps, eps]

####################### FUNCTIONS #####################################

def LinW(a, b):
	return np.concatenate((np.ravel(a), np.ravel(b)))

# Unlinearize AutoEncoder data: Take a vector, break it into two vectors, and roll it back up
def unLinWAllAE(vec):	
	W1 = np.asarray([vec[0			: f2*f1]])
	W2 = np.asarray([vec[f2*f1 		: f2*f1*2]])
	b1 = np.asarray([vec[f2*f1*2 	: f2*f1*2 + f2]])
	b2 = np.asarray([vec[ f2*f1*2 + f2 : f2*f1*2 + f2 + f1]])
	return W1.reshape(f2, f1) , W2.reshape(f1, f2), b1.reshape(f2, 1), b2.reshape(f1, 1)

# Unlinearize SOFT data: Take a vector, break it into two vectors, and roll it back up
def unLinW1(vec):	
	W1 = np.asarray([vec[0		: f2*f1]])
	b1 = np.asarray([vec[f2*f1	:]])
	return W1.reshape(f2, f1) , b1.reshape(f2, 1)
def unLinW2(vec):	
	W2 = np.asarray([vec[0		: f3*f2]])
	b2 = np.asarray([vec[f3*f2	:]])
	return W2.reshape(f3, f2) , b2.reshape(f3, 1)


# Calculate the Hypothesis (for layer l to l+1)
def hypothesis(W, b, dat):
	Max = np.amax(np.matmul(W, dat.T) + b)
	numer = np.exp( np.matmul(W, dat.T) + b - Max )	# 200 x 15298 for W1, b1
	denom = np.asarray([np.sum(numer, axis=0)])
	return (numer/denom).T


# Calculate the Hypothesis (layer 3) using just layer 1.
def ForwardProp(WA1, WA2, a1):
	W1, b1 = unLinW1(WA1)
	W2, b2 = unLinW2(WA2)
	# Calculate a2 (g.m x 200)
	a2 = hypothesis(W1, b1, a1)
	# Calculate and return the output from a2 and W2 (g.m x 10)
	a3 = hypothesis(W2, b2, a2)
	return a2, a3

# Calculate the regularized Cost J(theta)
def RegJCost(WA2, WA1, a1, ymat):
	# Forward Propagate
	a2, a3 = ForwardProp(WA1, WA2, a1)
	# Seperate and reshape the Theta values
	W2, b2 = unLinW2(WA2)
	# Calculate J(W, b). ymat and a3 are the same shape: 15298 x 10
	return (-1.0 / len(y))*np.sum( np.multiply(np.log(a3), ymat)  ) + lamb*0.5*np.sum(W2**2)


def BackProp(WA2, WA1, a1, ymat):
	# To keep track of how many times this code is called
	global gStep
	gStep += 1
	if gStep % 50 == 0:
		print 'Global Step: ', gStep, 'with JCost: ',  RegJCost(WA2, WA1, a1, ymat)
	if gStep % 200 == 0:
		print 'Saving Global Step : ', gStep
		saveW(WA2)

	# Forward Propagate
	a2, a3 = ForwardProp(WA1, WA2, a1)	# a2 (g.m x 200), a3 (g.m x 10)
	# Seperate and reshape the W and b values
	W2, b2 = unLinW2(WA2)
	
	# Now, to get backprop to work, I had to remake the theta matrices we had previously. Sandwich b2 onto W2
	WAll2 = np.hstack((b2, W2))
	# Attach a column of 1's onto a2
	left = np.array([[1] for i in range(len(col(ymat, 0))) ])
	a2ones = np.hstack((left, a2))
	# Calculate the derivative for both W2 and b2 at the same time
	DeltaWAll2 = (-1.0 / len(y))*np.matmul((ymat - a3).T, a2ones) + lamb*WAll2		# (g.f3, g.f2)
	# Seperate these back into W2 and b2 and linearize it
	return LinW(DeltaWAll2[:,1:], DeltaWAll2[:,:1])


def Norm(mat):
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0.00001
	nMax = 0.99999
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin

# Generate the y-matrix. This is called only once, so I use loops
def GenYMat(yvals):
	yvals = np.ravel(yvals)
	yArr = np.zeros((len(yvals), 10))
	for i in range(len(yvals)):
		for j in range(10):
			if yvals[i] == j:
				yArr[i][j] = 1
	return yArr

############################## MAIN CODE ################################
dat, y = PrepData('09')


dat = dat[:m, :]	# len(y)/2 For 04, 59 testing
y = y[:m]


file_name = 'output_folder/optimalweights_lambda_' + str(lamb) + '_Beta_' + str(beta) + '_Rho_' + str(rho) + '.out' #This is the name of the file where we pull our theta weights from

best_thetas = np.genfromtxt(file_name,dtype = 'float')
print(best_thetas.shape)

W1, W2AE, b1, b2AE = unLinWAllAE(best_thetas)	# W1: 200 x 784, b1: 200 x 1
W2 = randMat(f3, f2)			# 10 x 200
b2 = randMat(f3, 1)				# 10 x 1
WA1 = LinW(W1, b1)	# 1D vector, probably length 157000
WA2 = LinW(W2, b2)	# 1D vector, probably length 2010



ymat = GenYMat(y)

print 'Initial W JCost: ', RegJCost(WA2, WA1, dat, ymat) 
# Check the gradient. Go up and uncomment the import check_grad to use. ~2.378977939526638e-05 for m=98 for randomized Ws and bs
print scipy.optimize.check_grad(RegJCost, BackProp, WA2, WA1, dat, ymat)

# Calculate the best theta values for a given j and store them. Usually tol=10e-4. usually 'CG'


res = minimize(fun=RegJCost, x0= WA2, method='L-BFGS-B', tol=1e-4, jac=BackProp, args=(WA1, dat, ymat) ) # options = {'disp':True}
bestWA2 = res.x

print 'Final W JCost', RegJCost(bestWA2, WA1, dat, ymat) 


