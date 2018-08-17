#Created by: Nick Kyriacou
#Created on: 8/16/2018
#Purpose: This attempts to recreate the idea of combining a strong and weak classifier to make predictions. The strong classifier is raised to a very high power so that only the most probable events remain as signals and everything else just remains as "noise"


from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
import xgboost as xgb
import pandas as pd
import numpy as np






def add_features(data_frame):
	
	#Some constants we can use later on: 
	muon_mass = 105.6583715 # Muon Mass (in MeV)
	c = 299.792458 # Speed of light (km/s)
	tau_mass = 1776.82 #Tau Mass (in MeV)
	data_frame_size = len(data_frame)

	#This function will install a bunch of engineered features onto our model
	data_frame["FlightDistanceSig"] = data_frame['FlightDistance']/data_frame['FlightDistanceError']
	data_frame['isolation_min'] = data_frame.loc[:,['isolationa','isolationb','isolationc','isolationd','isolatione','isolationf']].min(axis=1)
        data_frame['IsoBDT_min'] = data_frame.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
        data_frame['track_Chi2Dof_MAX'] = data_frame.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
        #data_frame['IP_sum'] = data_frame.loc[:,['p0_IP','p1_IP','p2_IP']].min(axis=1)
        #data_frame['IPSig_sum'] = data_frame.loc[:,['p0_IPSig','p1_IPSig','p2_IPSig']].max(axis=1)
	data_frame['CDF_min']  = data_frame.loc[:,['CDF1','CDF2','CDF3']].min(axis=1)
	data_frame['DOCA_MAX'] = data_frame.loc[:,['DOCAone','DOCAtwo','DOCAthree']].max(axis = 1)
	data_frame['Speed']=data_frame['FlightDistance']/data_frame['LifeTime']
    	data_frame['flight_dist_sig2'] = (data_frame['FlightDistance']/data_frame['FlightDistanceError'])**2
	return(data_frame)


train = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

print('Train and Test data successfully read in')

train = add_features(train)
test = add_features(test)


train = train[ train['min_ANNmuon'] > 0.4 ] 
print('Filtered out events with min_ANNmuon > 0.4')


labels = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal','SPDhits']
features = list(f for f in train.columns if f not in labels)
num_features = train[features].shape[1]

print('Successfully added features')

#First we should pre-process our data
scaler = StandardScaler().fit(train[features].values)
X = scaler.transform(train[features].values)
Y = np_utils.to_categorical(train['signal'].values)
test = scaler.transform(test[features].values)
print('Inputs and Test data Pre-Processed')


#Now let's build our keras NN model.


num_epochs = 150
# deep pyramidal MLP, narrowing with depth
model_1 = Sequential()

model_1.add(Dense(800,input_shape = (num_features,)))
model_1.add(PReLU())

model_1.add(Dropout(0.5))
model_1.add(Dense(2))

model_1.add(Activation('softmax'))
model_1.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model_1.fit(X, Y, batch_size=128, nb_epoch=num_epochs, validation_data=None, verbose=2)

print('NN trained')

NN_guesses_1 = model_1.predict(test, batch_size=256, verbose=0)[:, 1]

print('NN Predictions made')
#Now lets create our xgboost model

print('Now setting parameters for our Extreme Gradient Boosting (Boosted Decision Trees) Model')

parameters = {'objective': 'binary:logistic','eta': 0.05, 'max_depth': 4, 'scale_pos_weight':5.0,'silent':1, 'seed':1}

tree_size = 400

#This first model is our geometric classifier
Train_DF = xgb.DMatrix(train[features],train["signal"])
xtreme_boosting_model = xgb.train(parameters,Train_DF,tree_size)
print('Successfully trained XGB classifier')


#Now let's make predictions for both our NN and our XGBoosting model!

test = pd.read_csv('data/test.csv')
test = add_features(test)

Test_DF = xgb.DMatrix(test[features])
test_predictions_xgb = xtreme_boosting_model.predict(Test_DF)

weight = 0.5

test_preds_weighted = weight*(NN_guesses_1**256) + (1.0 - weight)*(test_predictions_xgb)
print('Successfully made predictions!')
for i in range(len(test)):
	if(test_preds_weighted[i] < 0):
		test_preds_weighted[i] = 0
	if(test_preds_weighted[i] > 1):
		test_preds_weighted[i] = 1

print('Creating Submission file')
submission_file = pd.DataFrame({'id': test['id'], 'prediction': test_preds_weighted})
submission_file.to_csv('Submission_Folder/Strong_Weak_XGB_NN_0.5_NN_exp.csv',index = False)
print('Successfully submitted')

