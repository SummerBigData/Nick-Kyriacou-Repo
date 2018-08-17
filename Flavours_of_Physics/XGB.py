#Purpose: The purpose of this script is to implement XGB (xtreme gradient boosting) as suggested on discussion forums. This is just a baseline XGBoost model.
#Created By: Nicholas Kyriacou
#Created on: 7/26/2018


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
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
Train_Data_Frame = xgb.DMatrix(train[features_filtered],train["signal"])
xtreme_boosting_model = xgb.train(parameters,Train_Data_Frame,tree_size)


print('model successfully trained')

print('now we can use this model to make predictions on our test set')

test_guesses = xtreme_boosting_model.predict(xgb.DMatrix(test[features_filtered]))
print(test_guesses)

print('Making submission file')

submission_file = pd.DataFrame({'id': test['id'], 'prediction':test_guesses})
submission_file.to_csv('Submission_Folder/xgboost_only_submission_file.csv',index = False)




#Let's make a plot to determine feature importance!

sns.set(font_scale = 0.50)
xgb.plot_importance(xgb.train(parameters,Train_Data_Frame,tree_size))
plt.show()
'''
Now that we can see what features are important for our model. This shows us what features are doing the most to drive the splits for most trees and where we may be able to make some improvements in feature engineering.
Question to ask PhD's is about whether or not these features that are deemed important are hard to replicate in a simulation of particle physics events. Because with a submission of XGBoost alone I fail to pass the Correlation agreement test. 
'''

#Now that we have submitted on the full training set, lets retrain on a lesser portion of the train set and test on the test set




print('lets test the accuracy of our classifier on a validation set')
#This small subset of code will split our training sample and test the accuracy of our model on the test set. 
X_train,X_test, y_train, y_test = train_test_split(train[features_filtered],train['signal'],test_size = 0.2,random_state = 1)
print('we succesfully split our data')

#Now we remake this into a dataframe
X_train = pd.DataFrame(X_train,columns = features_filtered)
X_test = pd.DataFrame(X_test,columns = features_filtered)

Train_Data_Frame_new = xgb.DMatrix(X_train[features_filtered],y_train)
xtreme_boosting_model = xgb.train(parameters,Train_Data_Frame_new,tree_size)

#Let's set our threshold to 0.5 (default by using round function)

CV_guesses = xtreme_boosting_model.predict(xgb.DMatrix(X_test))
predictions = [round(guess) for guess in CV_guesses]

accuracy = accuracy_score(y_test,predictions)
print('Accuracy:  ' ,(accuracy*100.0))




