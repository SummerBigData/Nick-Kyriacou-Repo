#This Script will calculate some of the highest correlations between the features and signal and production. It will also create histograms to manually examine some of the distributions of our various variables. This is incredibly useful for visualization the seperation between my signal and background (simulated and real data)
#Created By: Nicholas Kyriacou
#Created on: 8/2/2018


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from hep_ml.gradientboosting import UGradientBoostingClassifier
from hep_ml.losses import BinFlatnessLossFunction

#Function definitions


def show_correlation(corr_data, corr_feature_name, n_most_correlated=15):
    corr=corr_data.corr().abs()
    most_correlated_feature=corr[corr_feature_name].sort_values(ascending=False)[:n_most_correlated].drop(corr_feature_name)
    most_correlated_feature_name=most_correlated_feature.index.values
    
    f, ax = plt.subplots(figsize=(30, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=most_correlated_feature_name, y=most_correlated_feature)
    plt.title("Various Features most correlated with {}".format(corr_feature_name))
    plt.show()
'''
def correlation_plot(data_frame, corr_var,num_features):
	#num_var = the number of most correlated variables we want to see
	#corr_var what is the variable we are looking for correlations with

	correlations = data_frame.corr().abs() #absolute value of features
	#Now we want to limit this to num_features length
	correlated_features = data_frame[corr_var].sort_values(ascending = False)[:num_features]
	most_correlated_feature_name = correlated_features.index.values
	fig,axis = plt.subplots(figsize = (30,12))
	plt.xticks(rotation = '90') #Rotates xlabels by 90 degrees
	sns.barplot(x = corr_var,y = correlated_features)
	plt.title('Signal as a function of correlation'.format(corr_var))
	plt.show()
'''

#First let's load training and testing dataframes
train = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

#Next we need to filter our dataset with features we don't want

show_correlation(corr_data = train, n_most_correlated = 20, corr_feature_name = 'signal')

#Because we can see that production is ~100% correlated with signal let's focus on what variables are highly correlated with production

show_correlation(corr_data = train,n_most_correlated = 20,corr_feature_name = 'production')

#Now let's make some histograms showing the discrepancy between various signal vs background features

#dividing our data into signal versus non signal features
signal = train[train.signal==1]
background = train[train.signal==0]

'''
plt.hist(signal.VertexChi2, bins = 150,label = 'Signal',alpha = 0.6,normed = True)
plt.hist(background.VertexChi2,bins = 150,label = 'Background',alpha = 0.6, normed = True)
plt.xlabel(r'Vertex $\chi^2$')
plt.ylabel('Counts (Fraction of 1)')
plt.legend()
plt.show()


plt.hist(signal.IP, bins = 150,label = 'Signal',alpha = 0.6,normed = True)
plt.hist(background.IP,bins = 150,label = 'Background',alpha = 0.6, normed = True)
plt.xlabel('Impact Parameter')
plt.ylabel('Counts (Fraction of 1)')
plt.legend()
plt.show()
'''
plt.hist(signal.min_ANNmuon, bins = 150,label = 'Signal',alpha = 0.6,normed = True)
plt.hist(background.min_ANNmuon,bins = 150,label = 'Background',alpha = 0.6, normed = True)
plt.xlabel('Minimum ANN of muon particles')
plt.ylabel('Counts (Fraction of 1)')
plt.legend()
plt.show()


plt.hist(signal.IPSig, bins = 20,label = 'Signal',alpha = 0.6,normed = True)
plt.hist(background.IPSig,bins = 150,label = 'Background',alpha = 0.6, normed = True)
plt.xlabel('Impact Parameter Significance')
plt.ylabel('Counts (Fraction of 1)')
plt.legend()
plt.show()


plt.hist(signal.ISO_SumBDT, bins = 75,label = 'Signal',alpha = 0.6,normed = True)
plt.hist(background.ISO_SumBDT,bins = 150,label = 'Background',alpha = 0.6, normed = True)
plt.xlabel('ISO_SumDBT (Track Isolation Variable)')
plt.ylabel('Counts (Fraction of 1)')
plt.legend()
plt.show()


plt.hist(signal.p0_track_Chi2Dof, bins = 20,label = 'Signal',alpha = 0.6,normed = True)
plt.hist(background.p0_track_Chi2Dof,bins = 150,label = 'Background',alpha = 0.6, normed = True)
plt.xlabel(r'p0 track $\chi^2$  Degrees of Freedom')
plt.ylabel('Counts (Fraction of 1)')
plt.legend()
plt.show()


plt.hist(signal.isolatione, bins = 20,range = (0.0,6.0),label = 'Signal',alpha = 0.6,normed = True)
plt.hist(background.isolatione,bins = 20,range = (0.0,6.0),label = 'Background',alpha = 0.6, normed = True)
plt.xlabel('Isolation_e (Track Isolation Variable')
plt.ylabel('Counts (Fraction of 1)')
plt.legend()
plt.show()

plt.hist(signal.iso, bins = 20,range = (0,20), label = 'Signal',alpha = 0.6,normed = True)
plt.hist(background.iso,bins = 20,range = (0,20),label = 'Background',alpha = 0.6, normed = True)
plt.xlabel('Iso (Track Isolation Variable')
plt.ylabel('Counts (Fraction of 1)')
plt.legend()
plt.show()







