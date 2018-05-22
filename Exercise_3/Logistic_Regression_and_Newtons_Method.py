# Import packages

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

#Defining all the functions we need
def logarithm_calc(input_1):
	for i in range(len(input_1)):
		input_1[i,0] = math.log(input_1[i,0])
		return(input_1)
def exponentiation(input_1):  #This function exponentiates each element in our array
	for i in range(len(input_1.T)):
		input_1[0,i] = math.exp(input_1[0,i])
	return(input_1)
def hypo(theta,inputs):
	exponent = -1.0*np.dot(theta.T,inputs)
	exponent = exponentiation(exponent)
	for i in range(len(exponent.T)):
		exponent[0,i] = 1.0/(float(exponent[0,i]+1.0))
	return(exponent)
def grad(hypothesis,outputs,inputs,total):
	subtraction =  hypothesis - outputs.T
	x = np.dot(subtraction,inputs)
	return(x/total)
def cost(hypothesis,outputs,total):
	print('shapes!')
	first = -1.0*np.dot(outputs.T,np.log(hypothesis.T))	
	print(first.shape)
	print(outputs)
	second = (1.0 - outputs)
	print(second)
	print(second.shape)
	third = np.log(1.0 - hypothesis)
	print(third.shape)
	combination = first - np.dot(second.T,third.T)
	print(combination.shape)
	return(combination/float(total))
	
 	


def Hessian(hypothesis,inputs,total):
	print('hypothesis')	
	print(hypothesis.shape)
	first  = np.zeros((1,80),float)
	H = np.zeros((3,3),float)
	for i in range(int(total)):
		first[0,i] = hypothesis[0,i]*(1.0-hypothesis[0,i])
		second = np.outer(inputs[i],inputs[i])
		H = H+(first[0,i]*second)
	H = (1.0/(float(total)))*H
	return(H)


#Loading data 
m = 80.0 #80 sets of training examples
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


test_one_int = np.hstack((x_ones,test_one))
test_two_int = np.hstack((x_ones,test_two))

#Now to make the first plot we should create a set of arrays that indicate the values of the first and second test scores when y = 1 and when y = 0
y_0_1 = []
y_0_2 = []
y_1_1 = []
y_1_2 = []
i = 0
for i in range(int(m)):
	if (y_original[i] == 0):
		y_0_1 = np.append(y_0_1,x_original_intercept[i,1])
		y_0_2 = np.append(y_0_2,x_original_intercept[i,2])
	else:
		y_1_1 = np.append(y_1_1,x_original_intercept[i,1])
		y_1_2 = np.append(y_1_2,x_original_intercept[i,2])
#Now let's reshape these lists into a usable array
y_0_1 = np.reshape(y_0_1,(len(y_0_1),1))
y_0_2 = np.reshape(y_0_2,(len(y_0_2),1))
y_1_1 = np.reshape(y_1_1,(len(y_1_1),1))
y_1_2 = np.reshape(y_1_2,(len(y_1_2),1))

#Now let's combine this set of arrays into the matrices of test scores for students that passed versus those that didn't
failed = np.hstack((y_0_1,y_0_2))
passed = np.hstack((y_1_1,y_1_2))

plt.plot(passed[:,0],passed[:,1],'ro')
plt.plot(failed[:,0],failed[:,1],'go' )
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Exam Scores')
plt.legend(['Accepted','Rejected'])
plt.show()



#Now let us minimize our cost function
i = 0.0
iteration_num = 20
theta = np.zeros((3,1),float)
temp_theta = np.zeros((3,1),float)

J_theta = np.zeros((iteration_num,1),float) 
for i in range(iteration_num):
	hypothesis = hypo(theta,x_original_intercept.T)
	J_theta[i] = cost(hypothesis,y_original,m)
	H = Hessian(hypothesis,x_original_intercept,m)
	print(H)
	gradient_J =  grad(hypothesis,y_original,x_original_intercept,m)
	temp_theta = theta - np.dot(np.linalg.inv(H),gradient_J.T)
	theta = temp_theta
print(theta)
plt.plot(J_theta)
plt.xlabel('Iteration Number')
plt.ylabel('Cost')
plt.title('Optimal Iteration Number')
plt.show()

#Now that we have found the optimal value of theta we can calculate and plot the decision boundary line
#First we want to find the left-most and right-most x values for Exam 1 Score
score_1_high = max(x_original[:,0])
score_1_low = min(x_original[:,0])
#Next find the corresponding y values ie, whether or not they were admitted into the college
exam_2_lowest_exam_1 = (-theta[0] - theta[1,0]*score_1_low)/(theta[2,0])
exam_2_highest_exam_1 = (-theta[0] - theta[1,0]*score_1_high)/(theta[2,0])
#Next we can convert these to a set of coordinate pairs
score_1_coords = [score_1_high,score_1_low]
score_2_coords = [exam_2_highest_exam_1, exam_2_lowest_exam_1]
print(theta[0,0])
print(theta[1,0])
print(theta[2,0])

plt.plot(passed[:,0],passed[:,1],'ro')
plt.plot(failed[:,0],failed[:,1],'go' )
plt.plot(score_1_coords,score_2_coords,'b')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Exam Scores')
plt.legend(["Accepted","Rejected","Decision Boundary"], prop = {'size':8})
plt.show()

#Calculating probability that a student with test scores 20 and 80 will get into a college 
student = [1,20,80]
student = np.reshape(student,(3,1))
was_he_admitted = hypo(theta,student)
print('Probability that the student will NOT be admitted into the college is ')
print(1.0-was_he_admitted[0,0])
