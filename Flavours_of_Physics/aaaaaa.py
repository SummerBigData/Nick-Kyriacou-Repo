#Purpose: This script will perform classification on the Flavours of Physics Kaggle Competition dataset using an xgboost model. To augment the model I will install a number of engineered features with which to train on. Some of these features were of my own invention, while other features I borrowed from looking at successful submissions on the discussion forums. 
#Created by: Nick Kyriacou
#Created on: 8/14/2018

#Importing Packages

import pandas as pd
import numpy as np
import xgboost as xgb
import evaluation

# List of the features for the first booster:
list1 = [
# Original features:
         'FlightDistance',
         'FlightDistanceError',
         'LifeTime',
         'IP',
         'IPSig',
         'VertexChi2',
         'dira',
         'pt',
         'DOCAone',
         'DOCAtwo',
         'DOCAthree',
         'IP_p0p2',
         'IP_p1p2',
         'isolationa',
         'isolationb',
         'isolationc',
         'isolationd',
         'isolatione',
         'isolationf',
         'iso',
         'CDF1',
         'CDF2',
         'CDF3',
         'ISO_SumBDT',
         'p0_IsoBDT',
         'p1_IsoBDT',
         'p2_IsoBDT',
         'p0_track_Chi2Dof',
         'p1_track_Chi2Dof',
         'p2_track_Chi2Dof',
         'p0_IP',
         'p0_IPSig',
         'p1_IP',
         'p1_IPSig',
         'p2_IP',
         'p2_IPSig',
# Extra features:
         'E',
         'FlightDistanceSig',
         'DOCA_sum',
         'isolation_sum',
         'IsoBDT_sum',
         'track_Chi2Dof',
         'IP_sum',
         'IPSig_sum',
         'CDF_sum'
        ]

# List of the features for the second booster:
list2 = [
# Original features:
         'dira',
         'pt',
         'p0_pt',
         'p0_p',
         'p0_eta',
         'p1_pt',
         'p1_p',
         'p1_eta',
         'p2_pt',
         'p2_p',
         'p2_eta',
# Extra features:
         'E',
         'pz',
         'beta',
         'gamma',
         'beta_gamma',
         'Delta_E',
         'Delta_M',
         'flag_M',
         'E0',
         'E1',
         'E2',
         'E0_ratio',
         'E1_ratio',
         'E2_ratio',
         'p0_pt_ratio',
         'p1_pt_ratio',
         'p2_pt_ratio',
         'eta_01',
         'eta_02',
         'eta_12',
         't_coll'
         ]




def add_features(df):
	# Physical constants:
	c = 299.792458     # Speed of light
	m_mu = 105.6583715 # Muon mass (in MeV)
	m_tau = 1776.82    # Tau mass (in MeV)


  
  	# Number of events:
  	N = len(df)
  
  	# Internal arrays:
  	p012_p = np.zeros(3)
  	p012_pt = np.zeros(3)
  	p012_z = np.zeros(3)
  	p012_eta = np.zeros(3)
  	p012_IsoBDT = np.zeros(3)
  	p012_track_Chi2Dof = np.zeros(3)
  	p012_IP = np.zeros(3)
  	p012_IPSig = np.zeros(3)
  	CDF123 = np.zeros(3)
 	isolation = np.zeros(6)
  
  	# Kinematic features related to the mother particle:
  	E = np.zeros(N)
  	pz = np.zeros(N)
  	beta = np.zeros(N)
  	gamma = np.zeros(N)
  	beta_gamma = np.zeros(N)
  	M_lt = np.zeros(N)
  	M_inv = np.zeros(N)
  	Delta_E = np.zeros(N)
  	Delta_M = np.zeros(N)
  	flag_M = np.zeros(N)
  
  	# Kinematic features related to the final-state particles p0, p1, and p2:
  	E012 = np.zeros((N,3))
  	E012_ratio = np.zeros((N,3))
  	p012_pt_ratio = np.zeros((N,3))
  	eta_01 = np.zeros(N)
  	eta_02 = np.zeros(N)
  	eta_12 = np.zeros(N)
  	t_coll = np.zeros(N)
  
  	# Other extra features:
  	FlightDistanceSig = np.zeros(N)
  	DOCA_sum = np.zeros(N)
  	isolation_sum = np.zeros(N)
  	IsoBDT_sum = np.zeros(N)
  	track_Chi2Dof = np.zeros(N)
  	IP_sum = np.zeros(N)
  	IPSig_sum = np.zeros(N)
  	CDF_sum = np.zeros(N)
  	
  	for i in range(N):
    		# Read some of the original features:  
    		pt = df['pt'].values[i]
    		dira = df['dira'].values[i]
    		LifeTime = df['LifeTime'].values[i]
    		FlightDistance = df['FlightDistance'].values[i]
    		FlightDistanceError = df['FlightDistanceError'].values[i]
    		DOCAone = df['DOCAone'].values[i]
    		DOCAtwo = df['DOCAtwo'].values[i]
    		DOCAthree = df['DOCAthree'].values[i]
    		isolation[0] = df['isolationa'].values[i]
    		isolation[1] = df['isolationb'].values[i]
    		isolation[2] = df['isolationc'].values[i]
    		isolation[3] = df['isolationd'].values[i]
    		isolation[4] = df['isolatione'].values[i]
    		isolation[5] = df['isolationf'].values[i]
    
    		for j in range(3):
      			p012_p[j] = df['p'+str(j)+'_p'].values[i]
      			p012_pt[j] = df['p'+str(j)+'_pt'].values[i]
      			p012_eta[j] = df['p'+str(j)+'_eta'].values[i]
      			p012_IsoBDT[j] = df['p'+str(j)+'_IsoBDT'].values[i]
      			p012_track_Chi2Dof[j] = df['p'+str(j)+'_track_Chi2Dof'].values[i]
      			p012_IP[j] = df['p'+str(j)+'_IP'].values[i]
      			p012_IPSig[j] = df['p'+str(j)+'_IPSig'].values[i]
      			CDF123[j] = df['CDF'+str(j+1)].values[i]
    
    		# Differences between pseudorapidities of the final-state particles:
    		eta_01[i] = p012_eta[0] - p012_eta[1]
    		eta_02[i] = p012_eta[0] - p012_eta[2]
    		eta_12[i] = p012_eta[1] - p012_eta[2]
    
    		# Transverse collinearity of the final-state particles (equals to 1 if they are collinear):
    		t_coll[i] = sum(p012_pt[:])/pt
    
    		# Longitudinal momenta of the final-state particles:
    		p012_z[:] = p012_pt[:]*np.sinh(p012_eta[:])
    
    		# Energies of the final-state particles:
    		E012[i,:] = np.sqrt(np.square(m_mu) + np.square(p012_p[:]))
    
    		# Energy and momenta of the mother particle:
    		E[i] = sum(E012[i,:])
    		pz[i] = sum(p012_z[:])
    		p = np.sqrt(np.square(pt) + np.square(pz[i]))
    
    		# Energies and momenta of the final-state particles relative to those of the mother particle:
    		E012_ratio[i,:] = E012[i,:]/E[i]
    		p012_pt_ratio[i,:] = p012_pt[:]/pt
    
    		# Mass of the mother particle calculated from FlightDistance and LifeTime:
    		beta_gamma[i] = FlightDistance/(LifeTime*c)
   		M_lt[i] = p/beta_gamma[i]
    
    		# If M_lt is around the tau mass then flag_M = 1 (otherwise 0):
    		if np.fabs(M_lt[i] - m_tau - 1.44) < 17: flag_M[i] = 1
    
    		# Invariant mass of the mother particle calculated from its energy and momentum:
    		M_inv[i] = np.sqrt(np.square(E[i]) - np.square(p))
    
    		# Relativistic gamma and beta of the mother particle:
    		gamma[i] = E[i]/M_inv[i]
    		beta[i] = np.sqrt(np.square(gamma[i]) - 1.)/gamma[i]
    
    		# Difference between M_lt and M_inv:
    		Delta_M[i] = M_lt[i] - M_inv[i]
    
    		# Difference between energies of the mother particle calculated in two different ways:
    		Delta_E[i] = np.sqrt(np.square(M_lt[i]) + np.square(p)) - E[i]
    
    		# Other extra features:
    		FlightDistanceSig[i] = FlightDistance/FlightDistanceError
    		DOCA_sum[i] = DOCAone + DOCAtwo + DOCAthree
    		isolation_sum[i] = sum(isolation[:])
    		IsoBDT_sum[i] = sum(p012_IsoBDT[:])
    		track_Chi2Dof[i] = np.sqrt(sum(np.square(p012_track_Chi2Dof[:] - 1.)))
    		IP_sum[i] = sum(p012_IP[:])
    		IPSig_sum[i] = sum(p012_IPSig[:])
    		CDF_sum[i] = sum(CDF123[:])
  
  	# Kinematic features related to the mother particle:
  	df['E'] = E
  	df['pz'] = pz
  	df['beta'] = beta
  	df['gamma'] = gamma
  	df['beta_gamma'] = beta_gamma
  	df['M_lt'] = M_lt
  	df['M_inv'] = M_inv
  	df['Delta_E'] = Delta_E
  	df['Delta_M'] = Delta_M
  	df['flag_M'] = flag_M
  
  	# Kinematic features related to the final-state particles:
  	df['E0'] = E012[:,0]
  	df['E1'] = E012[:,1]
  	df['E2'] = E012[:,2]
  	df['E0_ratio'] = E012_ratio[:,0]
  	df['E1_ratio'] = E012_ratio[:,1]
  	df['E2_ratio'] = E012_ratio[:,2]
  	df['p0_pt_ratio'] = p012_pt_ratio[:,0]
  	df['p1_pt_ratio'] = p012_pt_ratio[:,1]
  	df['p2_pt_ratio'] = p012_pt_ratio[:,2]
  	df['eta_01'] = eta_01
  	df['eta_02'] = eta_02
  	df['eta_12'] = eta_12
  	df['t_coll'] = t_coll
  	
  	# Other extra features:
  	df['FlightDistanceSig'] = FlightDistanceSig
  	df['DOCA_sum'] = DOCA_sum
  	df['isolation_sum'] = isolation_sum
  	df['IsoBDT_sum'] = IsoBDT_sum
  	df['track_Chi2Dof'] = track_Chi2Dof
  	df['IP_sum'] = IP_sum
  	df['IPSig_sum'] = IPSig_sum
  	df['CDF_sum'] = CDF_sum
  	
	return df

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
#geometric_base_features = ['FlightDistance','FlightDistanceError','LifeTime','IP','CDF1','CDF2','CDF3','IPSig','dira','DOCAone','DOCAtwo','DOCAthree','isolationa','isolationb','isolationc','isolationd','isolatione','isolationf','iso','pt','VertexChi2','IP_p0p2','IP_p1p2','ISO_SumBDT','p0_IsoBDT','p1_IsoBDT','p2_IsoBDT','p0_track_Chi2Dof','p1_track_Chi2Dof','p2_track_Chi2Dof','p0_IP','p1_IP','p2_IP','p0_IPSig','p1_IPSig','p2_IPSig']
#geometric_classifier_added_features = ['Energy','FlightDistanceSig','DOCA_sum','CDF_sum','IPSig_sum','IP_sum','track_Chi2Dof','IsoBDT_sum','isolation_sum']

#kinematic_base_features = ['dira','pt','p0_pt','p1_pt','p2_pt','p0_eta','p1_eta','p2_eta','p0_p','p1_p','p2_p']
#kinematic_classifier_added_features = ['Energy','Energy_0','Energy_1','Energy_2','Energy_total_Energy_0_ratio','Energy_total_Energy_1_ratio','Energy_total_Energy_2_ratio','momentum_transverse_p0_ratio','momentum_transverse_p1_ratio','momentum_transverse_p2_ratio','eta_01','eta_02','eta_12','p012_pt_sum_pt_ratio','pz','beta_gamma','Mass_Diff','Energy_Diff','gamma','beta','Mass_flag']

#geometric_classifier_features = geometric_base_features + geometric_classifier_added_features
#kinematic_classifier_features = kinematic_base_features + kinematic_classifier_added_features

print('geo')
#print(geometric_classifier_features)
print('kin')
#print(kinematic_classifier_features)
#Now let's train our dataset on this classifier.
labels = ['signal','mass','production','min_ANNmuon','SPDhits']
print('Removing features: ', labels)

#features = list(f for f in train.columns if f not in labels)

#We have two different classifiers that we want to train on.

#This first model is our geometric classifier
Train_geoDF = xgb.DMatrix(train[list1],train["signal"])
xtreme_boosting_model_geometric = xgb.train(parameters,Train_geoDF,tree_size_1)
print('Successfully trained geometric classifier')


#This second model is our kinematic classifer
Train_kinDF = xgb.DMatrix(train[list2],train["signal"])
xtreme_boosting_model_kinematic =  xgb.train(parameters,Train_kinDF, tree_size_2)
print('Successfully trained kinematic classifier')


Test_geoDF = xgb.DMatrix(test[list1])
Test_kinDF = xgb.DMatrix(test[list2])
geo_test_preds = xtreme_boosting_model_geometric.predict(Test_geoDF) #Geometric Predictions
kin_test_preds = xtreme_boosting_model_kinematic.predict(Test_kinDF) #Kinematic Predictions
print('Successfully made predictions')
weight = 0.80 #How much we are weighting our geometric classifer predictions
test_preds = 0.5*(geo_test_preds*weight + kin_test_preds*(1.0-weight))
print('Creating Submission file')
submission_file = pd.DataFrame({'id': test['id'], 'prediction': test_preds})
submission_file.to_csv('Submission_Folder/8.15.2018.3.csv',index = False)
print('Successfully submitted')

