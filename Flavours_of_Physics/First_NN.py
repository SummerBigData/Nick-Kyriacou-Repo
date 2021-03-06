#Purpose: This function is the first Neural Network that I made. It didn't perform so well because I created a hidden layer node that was impossibly large and didn't pre-scale the inputs as I should have.
#Created by: Nick Kyriacou
#Created on: 7/23/2018

#Importing Packages
import numpy as np
import pandas as pd
import keras 
import sklearn
from numpy import random 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
#Function Definitions

def Filter_Training_Data(x):
	labels = ['mass','production','min_ANNmuon','signal','id']
	#Next we remove these labels
	#However we also want to keep our list of id's because this helps us locate specific events as well as our labels (signal), so let's save these first and return them as well
	y = x['signal']
	input_num = x['id']
	x.drop(labels,axis = 1,inplace = True)
	data = x

	return(x,input_num,y)

def Filter_Testing_Data(x):
	labels = ['id']
	input_num = x['id']
	x.drop(labels,axis = 1,inplace =True)
	
	return(np.array(x),input_num)
#Main Code Starts Here

#First let's read in our training and testing data

training = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

print(training.shape)
print (test.shape)
#Let's see what information exists in our data sets
for column in training:
	print(column)
for column in test:
	print(column)
#As we can see there are four features missing in our training and testing sets
#The missing features are: mass (reconstructed T candidate mass), production (source of T), min_ANNmuon (Muon identification), signal (target variable we predict)
print('train')
print(list(training))
print('test')
print(list(test))

#Thus we want to remove these features from our training set because we wont have them for the testing set and are thus useless to classify on


train_trimmed,train_ids,train_labels = Filter_Training_Data(training)
train_trimmed = np.array(train_trimmed)
train_labels = np.array(train_labels)
train_ids = np.array(train_ids)

#Looking at the order of the output labels one can evidently see that the events are ordered first for background then signal. Thus we need to randomly shuffle them. Let's use a seed to get reproducible results
#We can only shuffle after reshaping as array because data is originally given to us as a list

np.random.seed(100)
np.random.shuffle(train_trimmed)
np.random.shuffle(train_labels)
np.random.shuffle(train_ids)

print(train_trimmed.shape)
print(train_ids.shape)
print(train_labels.shape)

#Because we are using categorical classification, we must have our true values (y) that we check our predictions against to be a (input_size,num_classes) matrix. 
#Numpy has a built in function that can do this for us.

train_labels = np_utils.to_categorical(train_labels)
print(train_labels.shape)

#Constant Definitions

length_training = 67553
length_testing = 855819
features = 46
nodes = 1000
batch_size = 128
num_classes = 2


#let's take a small sample just for testing purposes
train_trimmed = train_trimmed[0:20000]
train_labels = train_labels[0:20000]



#Pre-processing inputs 
scaled_down_train = MinMaxScaler()
train_trimmed = scaled_down_train.fit_transform(train_trimmed)

#Declaring our models structure
model = Sequential()

#Let's start with a simple neural network with one hidden layer

model.add(Dense(len(train_trimmed),input_dim = features, activation = 'relu'))
model.add(Dense(nodes,activation = 'relu'))
model.add(Dense(num_classes,activation = 'softmax')) #There are two different output classes
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop',metrics = ['accuracy'])
model.fit(train_trimmed,train_labels,epochs = 5,batch_size = batch_size,validation_split = 0.2)

#Let's Grab the testing set to make predictions on 

testing_data, test_ids = Filter_Testing_Data(test)

#We also need to seperately pre-process this test data because we also used pre-processing on our input data

scaled_down_test = MinMaxScaler()
testing_data = scaled_down_test.fit_transform(testing_data)

#Next let's make predictions for our test set
predictions = model.predict(testing_data,batch_size = 256)[:,1]
print('our testing predictions are: ')
print(predictions)

#Now let's prepare this script to create a submission file
submission_file = pd.DataFrame({'id': test_ids, "Prediction": predictions})
submission_file.to_csv("Keras_NN_v1.csv",index = False)




