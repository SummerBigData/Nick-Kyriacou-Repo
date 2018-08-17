#Created by: Nick Kyriacou
#Created on: 8/16/2018
#Purpose: This attempts to recreate the idea of combining a strong and weak classifier to make predictions. The strong classifier is raised to a very high power so that only the most probable events remain as signals and everything else just remains as "noise"


from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
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

scaler = StandardScaler().fit(train[features].values)
X = scaler.transform(train[features].values)
Y = np_utils.to_categorical(train['signal'].values)
test = scaler.transform(test[features].values)
print('Inputs and test data Pre-Processed')


#Now let's build our keras NN model. To ensure consistency we can average together the predictions of the keras NN run 5 seperate times.
num_epochs = 150
# deep pyramidal MLP, narrowing with depth
model_1 = Sequential()
#model.add(Dropout(0.13, input_shape=(X_train.shape[1],)))
model_1.add(Dense(800,input_shape = (num_features,)))
model_1.add(PReLU())

model_1.add(Dropout(0.5))
model_1.add(Dense(2))

model_1.add(Activation('softmax'))
model_1.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#Second Model
model_2 = Sequential()
model_2.add(Dense(800,input_shape = (num_features,)))
model_2.add(PReLU())

model_2.add(Dropout(0.5))
model_2.add(Dense(2))

model_2.add(Activation('softmax'))
model_2.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#Third Model
model_3 = Sequential()
model_3.add(Dense(800,input_shape = (num_features,)))
model_3.add(PReLU())

model_3.add(Dropout(0.5))
model_3.add(Dense(2))

model_3.add(Activation('softmax'))
model_3.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#Fourth Model
model_4 = Sequential()
model_4.add(Dense(800,input_shape = (num_features,)))
model_4.add(PReLU())

model_4.add(Dropout(0.5))
model_4.add(Dense(2))

model_4.add(Activation('softmax'))
model_4.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#Fifth Model
model_5 = Sequential()
model_5.add(Dense(800,input_shape = (num_features,)))
model_5.add(PReLU())

model_5.add(Dropout(0.5))
model_5.add(Dense(2))

model_5.add(Activation('softmax'))
model_5.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#Now let's train each of these models
print('Training NN 1')
model_1.fit(X, Y, batch_size=128, nb_epoch=num_epochs, validation_data=None, verbose=2)
print('Training NN 2')
model_2.fit(X, Y, batch_size=128, nb_epoch=num_epochs, validation_data=None, verbose=2)
print('Training NN 3')
model_3.fit(X, Y, batch_size=128, nb_epoch=num_epochs, validation_data=None, verbose=2)
print('Training NN 4')
model_4.fit(X, Y, batch_size=128, nb_epoch=num_epochs, validation_data=None, verbose=2)
print('Training NN 5')
model_5.fit(X, Y, batch_size=128, nb_epoch=num_epochs, validation_data=None, verbose=2)


print('all models successfully trained')


#Now make predictions for each model
print('Predicting Model 1')
test_guesses_1 = model_1.predict(test, batch_size=256, verbose=0)[:, 1]
print('Predicting Model 2')
test_guesses_2 = model_2.predict(test, batch_size=256, verbose=0)[:, 1]
print('Predicting Model 3')
test_guesses_3 = model_3.predict(test, batch_size=256, verbose=0)[:, 1]
print('Predicting Model 4')
test_guesses_4 = model_4.predict(test, batch_size=256, verbose=0)[:, 1]
print('Predicting Model 5')
test_guesses_5 = model_5.predict(test, batch_size=256, verbose=0)[:, 1]

#Averaging test guesses together
NN_test_guesses = (test_guesses_1 + test_guesses_2 + test_guesses_3 + test_guesses_4 + test_guesses_5)/5.0

# Forum idea of 'strong' + 'weak' classifier brought to a radical
# Raise predictions for NN (strong classifier) to a very high power, this effectively only keeps strong predictions, then just substitute the rest of my predictions with ('noise')

test = pd.read_csv('data/test.csv')

random_classifier = np.random.rand(len(NN_test_guesses))
q = 0.93
np.random.seed(1337) # for reproducibility
combined_probs = q * (NN_test_guesses ** 30) + (1 - q) * random_classifier
df = pd.DataFrame({"id": test['id'], "prediction": combined_probs})
df.to_csv("Submission_Folder/Strong_Weak_NN_150_epochs_30_.93.csv", index=False);
