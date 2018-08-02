#Purpose: The purpose of this script is to implement a weighted average of XGB (xtreme gradient boosting) and a RF classifier to help pass the CVM test. 
#Created By: Nicholas Kyriacou
#Created on: 7/30/2018


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#First let's load training and testing dataframes
train = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

#Next we need to filter our dataset with features we don't want

features_filtered = list(train.columns[1:-5])

#The features we removed were (id, min_ANNmuon,mass,signal (training labels), production, & SPDhits)
#Removing SPDhits helps us pass the correlation test

#Now let's train an XGBoost model

#First we should set the learning task parameters for our model.
#Thus let's initialize a list of parameters we will use to describe our model.
print('we are setting parameters for our XGB model')
parameters = { "objective": "binary:logistic", "eta": 0.4, "max_depth" : 6, "min_child_weight": 3, "silent":1,"subsample":0.7,"colsample_bytree":0.7,"seed":1}

tree_size = 2000
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
Train_Data_Frame = xgb.DMatrix(train[features_filtered],train["signal"])
xtreme_boosting_model = xgb.train(parameters,Train_Data_Frame,tree_size)


print('XGB model successfully trained')

print('Now we want to train a Randon Forest Model (RF)')

random_forest_model = RandomForestClassifier(n_estimators = 300, random_state = 1,criterion = 'entropy')
#n_estimator is the number of trees in our model
#criterion is the function that determines how to evaluate a split in our model

random_forest_model.fit(train[features_filtered],train["signal"])

print('RF model successfully trained')


print('Now we are going to make predictions on our test set')
random_forest_predictions = random_forest_model.predict_proba(test[features_filtered])[:,1]
print('predictions made for RF')
print('Now predicting for xgb')
Test_Data_Frame = xgb.DMatrix(test[features_filtered])
xtreme_boosting_predictions = xtreme_boosting_model.predict(Test_Data_Frame)
print('predictions made for xgb')

print('weighting test probabilities (equally)')

testing_guesses = random_forest_predictions*0.8 + xtreme_boosting_predictions*0.2

print('creating submission file')
submission_file = pd.DataFrame({'id': test['id'], 'prediction': testing_guesses})
submission_file.to_csv('Submission_Folder/RF_XGB_submission_file_1.csv',index = False)
print('Successfully submitted')
