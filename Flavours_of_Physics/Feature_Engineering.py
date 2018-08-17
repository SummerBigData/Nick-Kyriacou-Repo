#The purpose of this code is to implement some basic feature engineering and see how this improves my model.
#Created By: Nick Kyriacou
#Created on: 8/7/2018

#Importing Packages
import numpy as np
import pandas as pd
import evaluation
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.ensemble import RandomForestClassifier

def add_features(data_frame):
	#Here we will engineer some features that will hopefully improve our classification model
	
	data_frame["FlightDistanceSig"] = data_frame['FlightDistance']/data_frame['FlightDistanceError']
	#data_frame["FlightDistanceSig_Squared"] = (data_frame['FlightDistance']/data_frame['FlightDistanceError'])**2
	data_frame['isolation_min'] = data_frame.loc[:,['isolationa','isolationb','isolationc','isolationd','isolatione','isolationf']].min(axis=1)
	#data_frame['iso_bdt_p_min'] = data_frame.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
	#This grabs the maximum value of the isolation a through f for each event and stores that as a feature
	data_frame['DOCA_max'] = data_frame.loc[:,['DOCAone','DOCAtwo','DOCAthree']].max(axis = 1)
	data_frame['p_track_Chi2Dof_MAX'] = data_frame.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
	data_frame['CDF_MIN'] = data_frame.loc[:,['CDF1','CDF2','CDF3']].min(axis=1)
	#data_frame['IP/IP_p0p2'] = data_frame['IP']/data_frame['IP_p0p2']
	#data_frame['IP/IP_p1p2'] = data_frame['IP']/data_frame['IP_p1p2']

	data_frame['IP*dira'] = data_frame['IP']*data_frame['dira']
	#data_frame['Speed'] = data_frame['FlightDistance']/data_frame['LifeTime']



	#data_frame['p_eta_min'] = data_frame.loc[:,['p0_eta','p1_eta','p2_eta']].min(axis=1)
	#data_frame['p_IP_min'] = data_frame.loc[:,['p0_IP','p1_IP','p2_IP']].min(axis=1)
	#data_frame['p_IPSig_max'] = data_frame.loc[:,['p0_IPSig','p1_IPSig','p2_IPSig']].max(axis=1)
	#data_frame['p_pt_min'] = data_frame.loc[:,['p0_pt','p1_pt','p2_pt']].min(axis=1)
	#data_frame['p_min'] = data_frame.loc[:,['p0_p','p1_p','p2_p']].min(axis=1)
	return(data_frame)


#First load in data

training = pd.read_csv('data/training.csv')

#Now let's add some of our engineered features to our dataFrame
print('Adding extra features')

training = add_features(training)


#Now remove unwanted features
labels = ['signal','mass','production','min_ANNmuon','SPDhits']
print('Removing features: ', labels)

features = list(f for f in training.columns if f not in labels)

print(features)
#Create XGBoost model

print('Setting XGB parameters')

parameters = { "objective": "binary:logistic", "eta": 0.4, "max_depth" : 6, "min_child_weight": 3, "silent":1,"subsample":0.7,"colsample_bytree":0.7,"seed":1 }

tree_size = 350

#Create Random Forest Classifier

print('Creating RF model')

random_forest_model = RandomForestClassifier(n_estimators = 300,random_state = 1,criterion = 'entropy')

print('Now Training XGB model') 

Train_DF = xgb.DMatrix(training[features],training["signal"])
xtreme_boosting_model = xgb.train(parameters,Train_DF,tree_size)
print('XGB successfully trained')

print('Training RF model')
random_forest_model.fit(training[features],training["signal"])
print('Random Forest Successfully Trained')

#Now lets make our test set predictions to test our model
print('Now lets make predictions on the test set')

test = pd.read_csv('data/test.csv')

test = add_features(test)

print('Predicting for XGB')
Test_DF = xgb.DMatrix(test[features])
xtreme_boosting_predictions = xtreme_boosting_model.predict(Test_DF)
print('XGB predictions successful')

print('Predicting for RF')
random_forest_predictions = random_forest_model.predict_proba(test[features])[:,1]



print('weighting test probabililities')
test_preds = random_forest_predictions*0.8 + xtreme_boosting_predictions*0.2
print(test_preds.shape)
for x in range(len(test_preds)):
	if(test_preds[x] > 1):
		test_preds[x] = 0.99

print('creating submission file')
submission_file = pd.DataFrame({'id': test['id'], 'prediction': test_preds})
submission_file.to_csv('Submission_Folder/Feature_Engineering_2/FDSig_Isolation_Min_DOCA_Max_p_track_Chi2Dof_MAX_CDF_Min_IP*dira.csv',index = False)
print('Successfully submitted')
