
#Importing Packages
import matplotlib
#matplotlib.use('agg')
import numpy as np
import pandas as pd
import keras 
import sklearn
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from numpy import random 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
#Function Definitions

def plt_corr(dataframe,dim):
	'''
	Function plots a graphical correlation matrix for each pair of columns in the datafram
	Inputs: 
		dataframe = dataframe
		size: vertical and horizontal size of the plot
	'''
	corr = dataframe.corr()
	corr.style.background_gradient()
	figure,axis = plt.subplots(figsize = (dim,dim)) 
	
	plt.matshow(corr)
	plt.xticks(range(len(corr.columns)),corr.columns,rotation = 90)
	plt.yticks(range(len(corr.columns)),corr.columns)
	plt.colorbar()
	plt.show()


def plot_seaborn_heatmap(dataframe,dim):
	
	figure,axis = plt.subplots(figsize = (dim,dim))
	corr = dataframe.corr()
	sns.heatmap(corr,mask = np.zeros_like(corr,dtype = np.bool),cmap = sns.diverging_palette(220,10,as_cmap = True),square = True,ax = axis)
	plt.xticks(range(len(corr.columns)),corr.columns,rotation = 90)
	plt.yticks(range(len(corr.columns)),corr.columns,rotation = 0)
	plt.show()
	
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

#Making a correlation plot of all our inputs
plt_corr(training,51)

plot_seaborn_heatmap(training,51)

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
#train_trimmed = np.array(train_trimmed)
#train_labels = np.array(train_labels)
#train_ids = np.array(train_ids)



for column in train_trimmed:
	print(column)



labels = ['LifeTime','dira','FlightDistance','FlightDistanceError','IP','IPSig','VertexChi2','pt','DOCAone','DOCAtwo','DOCAthree','IP_p0p2','IP_p1p2','isolationa','isolationb','isolationc','isolationd','isolatione','isolationf','iso','CDF1','CDF2','CDF3','ISO_SumBDT','p0_IsoBDT','p1_IsoBDT','p2_IsoBDT','p0_track_Chi2Dof','p1_track_Chi2Dof','p2_track_Chi2Dof','p0_IP','p1_IP','p2_IP','p0_IPSig','p1_IPSig','p2_IPSig','p0_pt','p1_pt','p2_pt','p0_p','p1_p','p2_p','p0_eta','p1_eta','p2_eta','SPDhits']


print('lets take a quick look at mass (before pre-processing)')
training = pd.read_csv('data/training.csv')
mass = training['mass']
plt.hist(mass,bins = 10000)
plt.title('mass')
plt.savefig('Picture_Folder/Before/before_pre_processing'+'mass'+'.png')
plt.show()

print('now lets take a look-sie afterwards')
training = pd.read_csv('data/training.csv')
scaled_down_train = MinMaxScaler()
training = scaled_down_train.fit_transform(training)
print(training.shape)
print(training[:,49].shape)
mass = training[:,49]
plt.hist(mass,bins = 10000)
plt.title('mass')
plt.savefig('Picture_Folder/After/after_pre_processing'+'mass'+'.png')
plt.show()
#Let's comment this out because we only need to make these  plots once
#Let's grab each column individually
'''
feature_array = np.zeros((len(training),46))
for i in range(46):
	feature_array[:,i] = train_trimmed[labels[i]]
print('making plots before')
for j in range(42,46):
	plt.hist(feature_array[:,j],bins = 5000)
	plt.title(str(labels[j]))
	plt.savefig('Picture_Folder/Before/before_pre_processing'+str(labels[j])+'.png')
	print(j)
	#plt.show()
print('making plots after')
#Now we pre-process our inputs
scaled_down_train = MinMaxScaler()
train_trimmed = scaled_down_train.fit_transform(train_trimmed)
print('we just pre-processed')
feature_array = np.zeros((len(training),46))
for i in range(46):
	feature_array[:,i] = train_trimmed[:,i]
for j in range(42,46):
	plt.hist(feature_array[:,j],bins = 5000)
	plt.title(str(labels[j]))
	plt.savefig('Picture_Folder/after_pre_processing'+str(labels[j])+'.png')
	print(j)	
	#plt.show()
print('now we finished saving these pics')
'''
#Ignore this part too
'''
LifeTime = train_trimmed['LifeTime']
plt.hist(LifeTime,bins = 10000)
plt.show()

dira = train_trimmed['dira']
plt.hist(dira,bins = 10000)
FlightDistance = train_trimmed['FlightDistance']

FlightDistanceError
IP
IPSig
VertexChi2
pt
DOCAone
DOCAtwo
DOCAthree
IP_p0p2
IP_p1p2
isolationa
isolationb
isolationc
isolationd
isolatione
isolationf
iso
CDF1
CDF2
CDF3
ISO_SumBDT
p0_IsoBDT
p1_IsoBDT
p2_IsoBDT
p0_track_Chi2Dof
p1_track_Chi2Dof
p2_track_Chi2Dof
p0_IP
p1_IP
p2_IP
p0_IPSig
p1_IPSig
p2_IPSig
p0_pt
p1_pt
p2_pt
p0_p
p1_p
p2_p
p0_eta
p1_eta
p2_eta
SPDhits
'''
#Now let's make a correlation matrix to see if we can find features that are strongly correlated to each other
#Looking at the order of the output labels one can evidently see that the events are ordered first for background then signal. Thus we need to randomly shuffle them. Let's use a seed to get reproducible results

'''
#This one just doesn't give the best plot
plt.matshow(train_trimmed.corr())
plt.show()
'''
#plt_corr(train_trimmed,46)
#This plot looks really good!


# plot_seaborn_heatmap(train_trimmed,46) Possible if I could have rotated my axis
#We can only shuffle after reshaping as array because data is originally given to us as a list
'''
np.random.seed(100)
np.random.shuffle(train_trimmed)
np.random.shuffle(train_labels)
np.random.shuffle(train_ids)
'''
#train_trimmed = train_trimmed[0:5000]
#train_trimmed.plot(kind = 'density',subplots = True,sharex = False)
#plt.show()
