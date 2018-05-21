# Import packages

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

 
ages = np.genfromtxt('ex2x.dat')
heights = np.genfromtxt('ex2y.dat')

## Plotting Age vs Height
plt.plot(ages,heights,'bo')
plt.xlabel('Ages of Boys from 2-8')
plt.ylabel('Heights of Boys')
plt.title('Ages vs Heights')
plt.show()
## Adding 1 to the column of ages (for x-intercepts)
age_new = np.array([[1]*len(ages),ages])
print(age_new)

#Now implementing first round of gradient descent
alpha = 0.07
theta_0 = 0.
theta_1 = 0.
sumg = 0.0
for i in range (0,len(ages)):
    	h_theta =  theta_0 + theta_1*(age_new[1][i])
    	sumg = sumg + (h_theta - heights[i])*age_new[0][i]

temp_0 = theta_0 - (1.0/float((len(ages))))*alpha*(sumg)

sumg = 0.0
for x in range (0,len(ages)):
	h_theta =  theta_0 + theta_1*age_new[1][x]
	sumg = sumg + (h_theta - heights[x])*age_new[1][x]


temp_1 = theta_1 - (1.0/float((len(ages))))*alpha*(sumg)

theta_0 = temp_0 
theta_1 = temp_1

print(theta_0)
print(theta_1)

print("1500 Iterations")
#Now doing this again but running loop until convergence

alpha = 0.07
theta_0 = 0
theta_1 = 0
sumg = 0.0
z = 0
while z < 1500:
    for i in range (0,len(ages)):
        h_theta =  theta_0 + theta_1*(age_new[1][i])
        sumg = sumg + (h_theta - heights[i])*age_new[0][i]
    temp_0 = theta_0 - (1.0/float((len(ages))))*alpha*(sumg)
    sumg = 0.0
    for i in range (0,len(ages)):
        h_theta =  theta_0 + theta_1*age_new[1][i]
        sumg = sumg + (h_theta - heights[i])*age_new[1][i]

    temp_1 = theta_1 - (1.0/float((len(ages))))*alpha*(sumg)

    theta_0 = temp_0 
    theta_1 = temp_1
    z= z+1


print(theta_0)
print(theta_1)


#Now let's define our hypothesis function
def hypo(theta,inputs):
	hypothesis = np.dot(theta,inputs.T)
	return hypothesis

#Defining Cost Function
def cost(total,hypothesis,outputs):
	sumg = np.sum((hypothesis - outputs)**2)
	return(sumg/(2.0*float(total)))


#Now let's plot this hypothesis function with our final Theta parameters
inputs = np.linspace(2,8,100)
Y = .750113 + 0.06389*inputs
plt.scatter(ages,heights,color='k')
plt.plot(inputs,Y)
plt.xlabel('Ages')
plt.ylabel('Heights (meters)')
plt.title('Plot of Hypothesis Function')
plt.show()

#Now Let's make our contour plot
X = np.arange(-3,3,.06) #Theta 0 range of values
Y = np.arange(-1,1,.02) #Theta 1 range of values

J = np.zeros((len(X),len(X)))

for n in range(len(X)):
	for i in range(len(X)):
		h = (X[n]*np.ones(ages.shape)+Y[i]*ages-heights)
		J[n,i] = (1.0/(2.0*50))*np.sum(h**2)
X_0,Y_1 = np.meshgrid(X,Y)  #Creates grid of Theta 0 and Theta 1 values




#Now let's make a contour plot!
fig = plt.figure()
contour_plot = plt.contour(X_0,Y_1,J)
plt.xlabel('Theta_0')
plt.ylabel('Theta_1')
plt.title('Cost Function Contour')
fig.colorbar(contour_plot, shrink =0.5,aspect = 16)
plt.show()


fig4 = plt.figure()

#Now plotting 3D surface map
ax = Axes3D(plt.gcf())
surface = ax.plot_surface(X_0,Y_1,J, cmap = cm.coolwarm)
plt.xlabel('Theta_0')
plt.ylabel('Theta_1')
plt.title('Cost')
fig4.colorbar(surface,shrink = 0.6, aspect= 15)
plt.show()


##theta = np.hstack((X,Y))
##hypothesis = hypo(theta,age_new)
##J = cost(50,hypothesis,heights)


