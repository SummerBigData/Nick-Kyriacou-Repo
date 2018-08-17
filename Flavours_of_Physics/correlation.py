#This Script will calculate some of the highest correlations between the features and signal and production. It will also create histograms to manually examine some of the distributions of our various variables.
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




agreement = pd.read_csv('correlation_tests/check_agreement.csv')
correlation = pd.read_csv('correlation_tests/check_correlation.csv')


print('simulated vs real')
print(list(agreement))
print('mass agreement')
print(list(correlation))

