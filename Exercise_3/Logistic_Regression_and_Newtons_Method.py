# Import packages

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Loading data 
m = 80 #80 sets of training examples
x_original = np.genfromtxt('ex4x.dat')
y_original = np.genfromtxt('ex4y.dat')
y_original = np.reshape(y_original,(len(y_original),1))
x_ones = np.ones((len(x_original),1))
#Slicing arrays and adding x-intercepts
x_original_intercept = np.hstack((x_ones,x_original))
i = 0
test_scores_first = [0 for i in range(len(y_original))]
test_scores_first = np.array(test_scores_first)
test_scores_second = [0 for i in range(len(y_original))]
test_scores_second = np.array(test_scores_second)
for i in range(len(x_original)):
	test_scores_first[i] = x_original[i][0]
	test_scores_second[i] = x_original[i][1]

test_one = np.resize(test_scores_first,(80,1))
test_two = np.resize(test_scores_second,(80,1))
print(test_one.shape)
print(test_two.shape)
print(x_ones.shape)
test_one_int = np.hstack((x_ones,test_one))
test_two_int = np.hstack((x_ones,test_two))
print(test_one_int)
print(test_two_int)

#Now to make the first plot we should create a set of arrays that indicate the values of the first and second test scores when y = 1 and when y = 0
y_0_1 = []
y_0_2 = []
y_1_1 = []
y_1_2 = []
i = 0
print('starting')
for i in range(m):
	if (y_original[i] == 0):
		y_0_1 = np.append(y_0_1,x_original_intercept[i,1])
		y_0_2 = np.append(y_0_2,x_original_intercept[i,2])
		print('we have failed')
	else:
		y_1_1 = np.append(y_1_1,x_original_intercept[i,1])
		y_1_2 = np.append(y_1_2,x_original_intercept[i,2])
#Now let's reshape these lists into a usable array
y_0_1 = np.reshape(y_0_1,(len(y_0_1),1))
y_0_2 = np.reshape(y_0_2,(len(y_0_2),1))
y_1_1 = np.reshape(y_1_1,(len(y_1_1),1))
y_1_2 = np.reshape(y_1_2,(len(y_1_2),1))
print(y_0_1.shape)
print(y_0_2.shape)
print(y_1_1.shape)
print(y_1_2.shape)
#Now let's combine this set of arrays into the matrices of test scores for students that passed versus those that didn't
failed = np.hstack((y_0_1,y_0_2))
passed = np.hstack((y_1_1,y_1_2))
print('failed')
print(failed)
print('passed')
print(passed)
plt.plot(passed[:,0],passed[:,1],'ro')
plt.plot(failed[:,0],failed[:,1],'go' )
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Exam Scores')
plt.legend(['Accepted','Rejected'])
plt.show()

print(y_original)
print("length")
print(len(x_original))

