#Purpose: This script will combine XGB with a NN and output a weighted average of the two
#Created on: 7/31/2018
#Created by: Nick Kyriacou

#Importing Packages
import numpy as np
import pandas as pd
import keras 
import sklearn
from numpy import random 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


#First import training data


training = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

#Next lets filter out some unwanted features

features_filtered = list(training.columns[1:-5])


print('we are setting parameters for our XGB model')
parameters = { "objective": "binary:logistic", "eta": 0.4, "max_depth" : 6, "min_child_weight": 3, "silent":1,"subsample":0.7,"colsample_bytree":0.7,"seed":1}

tree_size = 250
''' This will describe what each parameter is doing for our model
#Here we are using logistic regression for binary classification
#Learning rate ('eta') of 0.4 
#max_depth of tree (this parameter can be tuned), however note that the larger the value the more likely overfitting is to occur.
#min_child_weight is the minimum sum of the weighted options needed to result in a new partition (child)
#silent = 1 means it won't print out messages
#subsample used to prevent overfitting, is the ratio of training instances that the model will random sample before growing trees
#colsample_bytree, this is the ratio of columns sampled when constructing each tree, sampling occurs once every time boosting iterations occur
#seed a random number seed
'''

#Now let's train the model

print('we are now training the Extreme Gradient Boosting model')

#In Order to train for xgb we must build DMatrices
scaled_down_train = MinMaxScaler()
train_trimmed = scaled_down_train.fit_transform(training[features_filtered])
Train_Data_Frame = xgb.DMatrix(train_trimmed,training["signal"])
xtreme_boosting_model = xgb.train(parameters,Train_Data_Frame,tree_size)


print('XGB model successfully trained')


print('Now lets train a neural network')

training = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

#Pre-processing our inputs
scale_down = MinMaxScaler()
x = scale_down.fit_transform(training[features_filtered])

model = Sequential()

#model.add(Dense((250),input_dim = 45,activation = 'relu'))
model.add(Dense(64,input_dim = 45,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

#x = training[features_filtered]
y = training['signal']

print(x.shape)
print(y.shape)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
model.fit(np.asarray(x),np.asarray(y),epochs = 50,batch_size = 128)

print('NN succesfully trained')

print('Now lets make predictions')
#First we should also scale down the test set
scaled_down_test = MinMaxScaler()
test_processed = scaled_down_test.fit_transform(test[features_filtered])
print('shape,' ,test_processed.shape)
Test_Data_Frame = xgb.DMatrix(test_processed)
xtreme_boosting_predictions = xtreme_boosting_model.predict(Test_Data_Frame)
predictions_NN = model.predict(np.asarray(test_processed),batch_size = 256)[:,0]

print('Now we will use weights to combine our NN and XGB predictions')


predictions_combined = xtreme_boosting_predictions*0.6599 + 0.3401*predictions_NN

print('creating submission file')
submission_file = pd.DataFrame({'id': test['id'], 'prediction': predictions_combined})
submission_file.to_csv('Submission_Folder/NN_XGB_submission_file.csv',index = False)
print('Successfully submitted')
