import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd


#This function scales the input array
def scale(array):
	mean = np.mean(array,axis = 0)
	std = np.std(array,axis = 0)
	return((array - mean)/std)

#Now let's define our hypothesis function
def hypo(theta,inputs):
	hypothesis = np.dot(theta,inputs.T)
	return hypothesis
#Defining Gradient Function
def grad(inputs,outputs,total,hypothesis):
	sumg = (np.dot((hypothesis - outputs),inputs)*(1.0/float(total)))
	return(sumg)
#Defining Cost Function
def cost(total,hypothesis,outputs):
	sumg = np.sum((hypothesis - outputs)**2)
	return(sumg/(2.0*float(total)))

#Let us find how much the cost is for each alpha value
def alpha_cost(iteration,theta,inputs,outputs,total):
	J_theta = np.zeros(iteration,float)
	alpha = .01 ##This also will change as well to pick a good learning rate

	for i in range (iteration): 
		hypothesis = hypo(theta,inputs)
		J_theta[i] = cost(total,hypothesis,outputs)
		theta = theta - (alpha)*grad(inputs,outputs,total,hypothesis)
	return(J_theta)


outputs = np.genfromtxt('ex3y.dat')
inputs = np.genfromtxt('ex3x.dat')
print(inputs.shape)
m = 47
#Defining Theta-vector
theta = np.zeros((1,3),float)
print (theta)
#print(inputs)
#print(outputs)

x_ones = np.ones((len(inputs),1))
#print(x_ones)
#print(x_ones.shape)
inputs_new = np.hstack((x_ones,inputs))


print(inputs_new)
print("Now scaling inputs")
#Scaling inputs
inputs_scaled = np.hstack((x_ones,scale(inputs)))
outputs = np.resize(outputs,(len(inputs_scaled),1))
print(outputs.shape)

print(inputs_scaled)



x = alpha_cost(50,theta,inputs_scaled,outputs,m)
plt.plot(x)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost of J')
plt.title('Cost of J for alpha = .01')
plt.show()


#Now let's do our linear regressions individually comparing Prices to Bedrooms and Prices to House Size
#First take individual outputs and make them all zero vectors
#I'm doing somewhat of a roundabout process for this so that I can get better practice with matrix computations and going back and forth between different data structures.
Bedrooms = [0 for i in range(len(outputs))]
HouseSize = [0 for i in range(len(outputs))]
scaled_first = scale(inputs)
for i in range(len(outputs)):
	Bedrooms[i] = scaled_first[i][1].tolist()
	HouseSize[i] = scaled_first[i][0].tolist()

#Now we have individual lists of each input scaled like we wanted to.
#Next we can create our lists of Theta and Temp Theta that we will use for our gradient descent method
theta_array = np.zeros((1,3),float)
temptheta_array =  np.zeros((1,3),float)
alpha = 1.0 #Change this to whatever you wnat

i = 0
for i in range(100):
	hypothesis = hypo(theta_array,inputs_scaled)
	temptheta_array[0] = theta_array[0] - alpha*(grad(inputs_scaled,outputs.T,47,hypothesis))
	

	theta_array[0] = temptheta_array[0]
	
print('Theta(0) = ', theta_array[0,0])
print('Theta(1) = ', theta_array[0,1])
print('Theta(2) = ', theta_array[0,2])

#Now let's make a plot with these values
beds = np.linspace(-3,3,100)
size = np.linspace(-2,4,100)
Y = 340412.6595 + 109447.796*size + -6578.35485*beds
plt.scatter(Bedrooms,outputs,color='k')
plt.plot(beds,Y,'r')
plt.xlabel('Bedrooms')
plt.ylabel('Prices (dollars)')
plt.title('Plot of Hypothesis Function')
plt.show()


Y = 340412.6595 + -6578.35485*beds + 109447.796*size 
plt.scatter(HouseSize,outputs,color = 'k')
plt.plot(size,Y,'r')
plt.xlabel('HouseSize (square feet)')
plt.ylabel('Prices ($)')
plt.title('Plot of Hypothesis Function')
plt.show()

#Now let's finish this with the closed form solution to a least squares fit
one = np.dot(inputs_new.T,inputs_new) ## (X^T * X)
two = np.linalg.inv(one) ## (X^T * X)^(-1)
three = np.dot(two,inputs_new.T) ## ((X^T * X)^(-1)) * X^T
four = np.dot(three,outputs) ## ((X^T * X)^(-1)) * X^T * Y
print(four)





