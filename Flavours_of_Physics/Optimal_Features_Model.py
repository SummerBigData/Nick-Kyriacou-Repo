#Purpose: This script will perform classification on the Flavours of Physics Kaggle Competition dataset using an xgboost model. To augment the model I will install a number of engineered features with which to train on. Some of these features were of my own invention, while other features I borrowed from looking at successful submissions on the discussion forums. 
#Created by: Nick Kyriacou
#Created on: 8/14/2018

#Importing Packages

import pandas as pd
import numpy as np
import xgboost as xgb
import evaluation
import matplotlib.pyplot as plt

#Function Definitions

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
        data_frame['IP_min'] = data_frame.loc[:,['p0_IP','p1_IP','p2_IP']].min(axis=1)
        data_frame['IPSig_max'] = data_frame.loc[:,['p0_IPSig','p1_IPSig','p2_IPSig']].max(axis=1)
	data_frame['CDF_min']  = data_frame.loc[:,['CDF1','CDF2','CDF3']].min(axis=1)
	data_frame['DOCA_MAX'] = data_frame.loc[:,['DOCAone','DOCAtwo','DOCAthree']].max(axis = 1)
	'''
        data_frame['FlightDistanceSig'] = data_frame['FlightDistance']/data_frame['FlightDistanceError']
        data_frame['DOCA_sum'] = data_frame['DOCAone'] + data_frame['DOCAtwo'] + data_frame['DOCAthree'] 
        data_frame['isolation_sum'] = data_frame['isolationa'] + data_frame['isolationb'] + data_frame['isolationc'] + data_frame['isolationd'] + data_frame['isolatione'] + data_frame['isolationf']
        data_frame['IsoBDT_sum'] = data_frame['p0_IsoBDT'] + data_frame['p1_IsoBDT'] + data_frame['p2_IsoBDT']
        data_frame['track_Chi2Dof'] = np.sqrt( np.square(data_frame['p0_track_Chi2Dof'] - 1.0) + np.square(data_frame['p1_track_Chi2Dof'] - 1.0) + np.square(data_frame['p2_track_Chi2Dof'] - 1.0) )
        data_frame['IP_sum'] = data_frame['p0_IP'] + data_frame['p1_IP'] + data_frame['p2_IP']
        data_frame['IPSig_sum'] = data_frame['p0_IPSig'] + data_frame['p1_IPSig'] + data_frame['p2_IPSig']
	data_frame['CDF_sum'] = data_frame['CDF1'] + data_frame['CDF2'] + data_frame['CDF3']
	'''
	#Here we calculate the energy of the final state particles 
	Energy_0 = np.sqrt( muon_mass**2 + np.square(data_frame['p0_p']) )
	Energy_1 = np.sqrt( muon_mass**2 + np.square(data_frame['p1_p']) )
	Energy_2 = np.sqrt( muon_mass**2 + np.square(data_frame['p2_p']) )	
	data_frame['Energy_0'] = Energy_0
  	data_frame['Energy_1'] = Energy_1
  	data_frame['Energy_2'] = Energy_2
	data_frame['Energy'] = Energy_0 + Energy_1 + Energy_2
	
	# The rest are features originating from Alexander Gramolin's code
  
  	# These are all kinematic features for each of the three final products after the decay event
  	data_frame['Energy_total_Energy_0_ratio'] = Energy_0 / ( Energy_0 + Energy_1 + Energy_2)
  	data_frame['Energy_total_Energy_1_ratio'] = Energy_1 / ( Energy_0 + Energy_1 + Energy_2)
  	data_frame['Energy_total_Energy_2_ratio'] = Energy_2 / ( Energy_0 + Energy_1 + Energy_2)
  	data_frame['momentum_transverse_p0_ratio'] = data_frame['p0_pt']/ (data_frame['p0_pt'] + data_frame['p1_pt'] + data_frame['p2_pt']) 
  	data_frame['momentum_transverse_p1_ratio'] = data_frame['p1_pt']/ (data_frame['p0_pt'] + data_frame['p1_pt'] + data_frame['p2_pt'])
  	data_frame['momentum_transverse_p2_ratio'] = data_frame['p2_pt']/ (data_frame['p0_pt'] + data_frame['p1_pt'] + data_frame['p2_pt'])

	#These next three features are the possible permutations of the difference in pseudorapidity of final products
  	data_frame['eta_01'] = data_frame['p0_eta'] - data_frame['p1_eta']
  	data_frame['eta_02'] = data_frame['p0_eta'] - data_frame['p2_eta']
  	data_frame['eta_12'] = data_frame['p1_eta'] - data_frame['p2_eta']
	#This feature is the Transverse collinearity of the final-state particles (value of 1 indicates collinear)
	data_frame['p012_pt_sum_pt_ratio'] = (data_frame['p0_pt'] + data_frame['p1_pt'] + data_frame['p2_pt']) / data_frame['pt']
	#was called 't_coll'

	# Kinematic features related to the mother particle:
	
	# We already found the energy of the T candidate, now lets calculate its momentum 

	# Momentum in the z (longitudinal) direction for each final product
	momentum_z_0 = data_frame['p0_pt']*np.sinh(data_frame['p0_eta'])
	momentum_z_1 = data_frame['p1_pt']*np.sinh(data_frame['p1_eta'])
	momentum_z_2 = data_frame['p2_pt']*np.sinh(data_frame['p2_eta'])
	data_frame['pz'] = momentum_z_0 + momentum_z_1 + momentum_z_2

  	
	#Two different ways to reconstruct the mass of the particle  	
	data_frame['beta_gamma'] = (data_frame['FlightDistance']/(data_frame['LifeTime']*c))
	p_initial_particle = np.sqrt(np.square(data_frame['pt']) + np.square(data_frame['pz']))
	data_frame['Mass_LifeTime_FlightDistance'] =   p_initial_particle/ data_frame['beta_gamma']
  	data_frame['Mass_invariant'] = np.sqrt( np.square(data_frame['Energy']) - np.square(p_initial_particle) )
 	data_frame['Mass_Diff'] = data_frame['Mass_LifeTime_FlightDistance'] - data_frame['Mass_invariant']
  	
	#For some reason is the Mass_LifeTime_FlightDistance is close to the tau mass then we raise a flag on the mass (otherwise it remains zero). Perhaps use this feature to help pass CvM test.
	Mass_flag = np.zeros(data_frame_size)	
	for i in range(data_frame_size):
		
		if np.fabs(data_frame['Mass_LifeTime_FlightDistance'].values[i] - tau_mass) < 18.44:
			Mass_flag[i] = 1	
	data_frame['Mass_flag'] = Mass_flag
	
	#This is the difference between the two seperate ways of calculating energy
	data_frame['Energy_Diff'] = np.sqrt(np.square(data_frame['Mass_LifeTime_FlightDistance']) + np.square(p_initial_particle)) - data_frame['Energy']
	
	#These two features are the gamme and beta features of the initial particle that we know from relavity
  	data_frame['gamma'] = data_frame['Energy'] / data_frame['Mass_invariant']	
	data_frame['beta'] = np.sqrt(np.square(data_frame['gamma']) - 1.0)/data_frame['gamma']
		
	return(data_frame)

#First let's read in our dataset

train = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

print('Train and Test data successfully read in')

'''
It is also recommended to only train on data that uses min_ANNmuon > 0.4 because only events in the test set with this characteristic are scored on. By filtering out these events we are eliminating a "dangerous background" decay D(+) -> K(-)pi(+)pi(+)
'''

train = train[ train['min_ANNmuon'] > 0.4 ] 
print('Filtered out events with min_ANNmuon > 0.4')

#Now let's add our engineered features onto our training set so we can begin classification

train = add_features(train)
test = add_features(test)

print('Successfully added features to training, testing, and correlation test sets!')

#Next lets build our XGBoost model.

print('Now setting parameters for our Extreme Gradient Boosting (Boosted Decision Trees) Model')

parameters = {'objective': 'binary:logistic','eta': 0.05, 'max_depth': 4, 'scale_pos_weight':5.0,'silent':1, 'seed':1}

tree_size_1 = 200
tree_size_2 = 100

'''
The following two lists will be lists of the features that we will use to train each of the classifier models on
'''
geometric_base_features = ['FlightDistance','FlightDistanceError','LifeTime','IP','CDF1','CDF2','CDF3','IPSig','dira','DOCAone','DOCAtwo','DOCAthree','isolationa','isolationb','isolationc','isolationd','isolatione','isolationf','iso','pt','VertexChi2','IP_p0p2','IP_p1p2','ISO_SumBDT','p0_IsoBDT','p1_IsoBDT','p2_IsoBDT','p0_track_Chi2Dof','p1_track_Chi2Dof','p2_track_Chi2Dof','p0_IP','p1_IP','p2_IP','p0_IPSig','p1_IPSig','p2_IPSig']
geometric_classifier_added_features = ['FlightDistanceSig','DOCA_MAX','CDF_min','track_Chi2Dof_MAX','IsoBDT_min','isolation_min','IPSig_max','IP_min']

kinematic_base_features = ['dira','pt','p0_pt','p1_pt','p2_pt','p0_eta','p1_eta','p2_eta','p0_p','p1_p','p2_p']
kinematic_classifier_added_features = ['Energy','Energy_0','Energy_1','Energy_2','Energy_total_Energy_0_ratio','Energy_total_Energy_1_ratio','Energy_total_Energy_2_ratio','momentum_transverse_p0_ratio','momentum_transverse_p1_ratio','momentum_transverse_p2_ratio','eta_01','eta_02','eta_12','p012_pt_sum_pt_ratio','pz','beta_gamma','Mass_Diff','Energy_Diff','gamma','beta','Mass_flag']

geometric_classifier_features = geometric_base_features + geometric_classifier_added_features
kinematic_classifier_features = kinematic_base_features + kinematic_classifier_added_features

print('geo')
print(geometric_classifier_features)
print('kin')
print(kinematic_classifier_features)
#Now let's train our dataset on this classifier.
labels = ['signal','mass','production','min_ANNmuon','SPDhits']
print('Removing features: ', labels)

#features = list(f for f in train.columns if f not in labels)

#We have two different classifiers that we want to train on.

#This first model is our geometric classifier
Train_geoDF = xgb.DMatrix(train[geometric_classifier_features],train["signal"])
xtreme_boosting_model_geometric = xgb.train(parameters,Train_geoDF,tree_size_1)
print('Successfully trained geometric classifier')


#This second model is our kinematic classifer
Train_kinDF = xgb.DMatrix(train[kinematic_classifier_features],train["signal"])
xtreme_boosting_model_kinematic =  xgb.train(parameters,Train_kinDF, tree_size_2)
print('Successfully trained kinematic classifier')


Test_geoDF = xgb.DMatrix(test[geometric_classifier_features])
Test_kinDF = xgb.DMatrix(test[kinematic_classifier_features])
geo_test_preds = xtreme_boosting_model_geometric.predict(Test_geoDF) #Geometric Predictions
kin_test_preds = xtreme_boosting_model_kinematic.predict(Test_kinDF) #Kinematic Predictions
print('Successfully made predictions')
weight = 0.93 #How much we are weighting our geometric classifer predictions
test_preds = geo_test_preds*weight + kin_test_preds*(1.0-weight)
print('Creating Submission file')
submission_file = pd.DataFrame({'id': test['id'], 'prediction': test_preds})
submission_file.to_csv('Submission_Folder/8.15.2018.2.csv',index = False)
print('Successfully submitted')


