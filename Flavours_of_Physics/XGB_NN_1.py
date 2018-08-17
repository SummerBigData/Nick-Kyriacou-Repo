#Purpose: This script will combine XGB with a NN and output a weighted average of the two. This script achieved much better results (0.89%) validation set accuracy but had trouble passing the KS test.
#Created on: 7/31/2018
#Created by: Nick Kyriacou

#Importing Packages
import numpy as np
import pandas as pd
import keras 
import sklearn
import evaluation
from numpy import random 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


#First import training data


training = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')
check_agreement = pd.read_csv('correlation_tests/check_agreement.csv')
#Next lets filter out some unwanted features

labels = ['signal','mass','production','min_ANNmuon','SPDhits','p1_track_Chi2Dof']
#features_filtered = list(training.columns[1:-5])
#extras = ['IP','IPSig']
#features_filtered = list(set(features_filtered) - set(extras))
features = list(f for f in training.columns if f not in labels)


#One dangerous background is a D(+)--> K(-)pi(+)pi(+) decay. This background can be eliminated by requiring that min_ANNmuon > 0.4
#training = training[training['min_ANNmuon'] > 0.4]
#Ignore above for now


print('we are setting parameters for our XGB model')
parameters = { "objective": "binary:logistic", "eta": 0.2, "max_depth" : 6, "min_child_weight": 3, "gamma":0.01, "silent":1,"subsample":0.7,"colsample_bytree":0.7,"seed":1}

tree_size = 575
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
#The following will scale down our inputs :( just checking something
scaled_down_train = MinMaxScaler()
x = scaled_down_train.fit_transform(training[features])

Train_Data_Frame = xgb.DMatrix(x,training["signal"])
xtreme_boosting_model = xgb.train(parameters,Train_Data_Frame,tree_size)


print('XGB model successfully trained')


print('Now lets train a neural network')

training = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

#Pre-processing our inputs again
scaled_down_train = MinMaxScaler()
x = scaled_down_train.fit_transform(training[features])


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
#First we need to seperately scale down the test set
scaled_down_test = MinMaxScaler()
test_scaled = scaled_down_test.fit_transform(test[features])
Test_Data_Frame = xgb.DMatrix(test_scaled)
xtreme_boosting_predictions = xtreme_boosting_model.predict(Test_Data_Frame)
predictions_NN = model.predict(np.asarray(test_scaled),batch_size = 256)[:,0]

print('Now we will use weights to combine our NN and XGB predictions')


predictions_combined = xtreme_boosting_predictions*0.1 + 0.9*predictions_NN
'''
print('Checking KS score')
agreement_probs_xgb = xtreme_boosting_model.predict(xgb.DMatrix(check_agreement[features]))
agreement_probs_NN = model.predict(np.asarray(check_agreement[features]),batch_size = 256)[:,0]

agreement_combined = 0.5*agreement_probs_xgb + 0.5*agreement_probs_NN
#First let's run our check_agreement.csv file and make predictions on those
ks = evaluation.compute_ks(agreement_combined[check_agreement['signal'].values == 0],agreement_combined[check_agreement['signal'].values ==1],check_agreement[check_agreement['signal'] == 0]['weight'].values,check_agreement[check_agreement['signal']==1]['weight'].values)
print('Features dropped are: ', labels)
print('KS metric 0.5 weights is:', ks)
'''
print('creating submission file')
submission_file = pd.DataFrame({'id': test['id'], 'prediction': predictions_combined})
submission_file.to_csv('Submission_Folder/NN_XGB_submission_file_SPD_p1_track_Chi2Dof.csv',index = False)
print('Successfully submitted')
