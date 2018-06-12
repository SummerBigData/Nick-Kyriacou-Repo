#This code visualizes the inputs and outputs generated from "Sparse_auto_encoder.py". It does this by grabbing the optimal thetas values previously calculated and then feeding forward twice with said theta values to generate output units and 10 input units. A Good Portion of this code was borrowed from Matt Kusz
#Owner: Nick Kyriacou
#Created: 6/11/2018

# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

####### Global variables #######
features = 64
length_hidden = 25
l = 100 #Lambda Parameter
Beta = 1.0 #Beta Parameter
P =  0.05#Rho Parameter
####### Definitions #######
# Sigmoid function

def sigmoid(W,B,inputs): #Add surens fix if there are overflow errors
	z = np.matmul(W, inputs.T) + B #Should be a 25x10000 matrix
	#print('in sigmoid')
	#print(z.shape)
	hypo = 1.0/(1.0 + np.exp(-1.0*z))
	return(hypo)


	
def seperate(theta_all): #This function will take a combined theta vector and seperate it into 4 of its specific components
	W_1 = np.reshape(theta_all[0:features*length_hidden],(length_hidden,features))
	B_1 = np.reshape(theta_all[2*features*length_hidden:2*features*length_hidden + length_hidden],(length_hidden,1))
	W_2 = np.reshape(theta_all[features*length_hidden:2*features*length_hidden],(features,length_hidden))
	B_2 = np.reshape(theta_all[2*features*length_hidden + length_hidden:len(theta_all)],(features,1))
	#Now that we have seperated each vector and reshaped it into usable format lets return each weight
	#print('in seperate')
	#print(W_1.shape)
	#print(W_2.shape)
	#print(B_1.shape)
	#print(B_2.shape)
	return(W_1,B_1,W_2,B_2)



# Feedforward
def feed_forward(theta_all,xvals):
	#First we must unravel our theta1 and theta2 and reshape them into the correct dimensions
	W_1,B_1, W_2, B_2 = seperate(theta_all)
	#print('inside feed forward')
	#print(W_1.shape)
	#print(B_1.shape)
	#print(xvals.shape)
	a_2 = sigmoid(W_1,B_1,xvals) #Calculates a_2 as a matrix
	a_2 = a_2.T
	#print(a_2.shape)
	#print('that was a_2')
	a_3 = sigmoid(W_2,B_2,a_2)
	a_3 = a_3.T
	#print(a_3.shape)
	#print('did this:feed_forward')
	return(a_3,a_2)


# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_hidden_size * global_visible_size], (global_hidden_size, global_visible_size))
	W2 = np.reshape(theta[global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size], (global_visible_size, global_hidden_size))
	b1 =np.reshape(theta[2 * global_hidden_size * global_visible_size: 2 * global_hidden_size * global_visible_size + global_hidden_size], (global_hidden_size, 1))
	b2 =np.reshape(theta[2 * global_hidden_size * global_visible_size + global_hidden_size: len(theta)], (global_visible_size, 1))
	
	return W1, W2, b1, b2


def Norm(mat): #Normalizes the data values (output or input) to something between 1 and 0
	Min = np.amin(mat)
	Max = np.amax(mat)
	nMin = 0
	nMax = 1
	return ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin


################################Main Code Starts Here###########################################
# Import the data (input) images we need
data = np.genfromtxt('output_folder/10000Random8x8.out',dtype = float)
data = np.reshape(data, (64, 10000))


# Import the optimal theta values we need
name = 'output_folder/optimalweights_l' + str(l) +'_Beta' + str(Beta) + '_Rho'+ str(P) +'.out' #everytime change the values to reflect which thetas you want
theta_final = np.genfromtxt(name,dtype = float)

# DATA PROCESSING


# Roll up data into matrix. Restrict it to [0,1]. Trim array to user defined size
data = np.asarray(data.reshape(10000,64))
# Normalize each image
for i in range(10000):
	data[i] = Norm(data[i])
a_1 = data[0:10000,:]




# FORWARD PROPAGATE AND CALCULATE BEST GUESSES
# Feed the best W and b vals into forward propagation
a_3, a_2 = feed_forward(theta_final, a_1)

for i in range(10000):
	a_2[i] = Norm(a_2[i])
	a_3[i] = Norm(a_3[i])


# Next step is to calculate the average deviation of outputs to inputs
abs_deviation = np.zeros((10000, 64))
for i in range(10000):
	abs_deviation[i] = np.abs(a_1[i]-a_3[i])

print 'The average seperation between a1 and a3 is (Note: 0-1, where 0 is close)', np.mean(abs_deviation)









# SHOW IMAGES
hspaceAll = np.asarray([ [0 for i in range(53)] for j in range(5)])
picAll = hspaceAll

for i in range(10):
	# Store the pictures
	picA1 = np.reshape(np.ravel(a_1[i*100]), (8,8))
	picA2 = np.reshape(np.ravel(a_2[i*100]), (5,5))
	picA3 = np.reshape(np.ravel(a_3[i*100]), (8,8))
#	print np.linalg.norm(a1[i*100])
#	print np.linalg.norm(a3[i*100])
	# DISPLAY PICTURES
	# To display a2 in revprop, a1, and a2 in forward prop, we design some spaces
	hspace = np.asarray([ [0 for i in range(8)] for j in range(8)])
	vspace1 = np.asarray([ [0 for i in range(5)] for j in range(1)])
	vspace2 = np.asarray([ [0 for i in range(5)] for j in range(2)])

	# We stitch the vertical spaces onto the pictures
	picA2All = np.concatenate((vspace1, picA2, vspace2), axis = 0)
	# We stitch the horizontal pictures together
	picAlli = np.concatenate((hspace, picA1, hspace, picA2All, hspace, picA3, hspace), axis = 1)
	# Finally, add this to the picAll
	picAll = np.vstack((picAll, picAlli, hspaceAll))

# Display the pictures
a = plt.figure()
plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.title('Lambda = '+ str(l) +' Beta = '  + str(Beta) +' Rho = ' + str(P) )
#plt.savefig('results/a123'+'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'rand'+g.randData+'.png',transparent=False, format='png')
plt.show()


'''
# We also want a picture of the activations for each node in the hidden layer
W1, W2, b1, b2 = unLinWAll(bestWAll)
W1Len = np.sum(W1**2)**(-0.5)
X = W1 / W1Len			# (25 x 64)
X = Norm(X)

picX = np.zeros((25,8,8))
for i in range(25):
	picX[i] = np.reshape(np.ravel(X[i]), (8,8))

hblack = np.asarray([ [1 for i in range(52)] for j in range(2)])
vblack = np.asarray([ [1 for i in range(2)] for j in range(8)])

picAll = hblack
for i in range(5):
	pici = np.concatenate((vblack, picX[5*i+0], vblack, picX[5*i+1], vblack, picX[5*i+2], vblack, picX[5*i+3], vblack, picX[5*i+4], vblack), axis = 1)
	picAll = np.vstack((picAll, pici, hblack))

# Display the pictures
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.savefig('results/aHL'+'Tol'+str(g.tolexp)+'Lamb'+str(g.lamb)+'rand'+g.randData+'.png',transparent=False, format='png')
plt.show()
'''
