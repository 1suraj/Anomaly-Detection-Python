# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time as t
import math
from datetime import datetime, date, time
from scipy import stats
from itertools import repeat


print("Initializing data...")
dataset = pd.read_csv('Outlier_Data.csv')
print("Initialized!")

features = ['Brain_Weight','Body_Weight']


#EllipticEnvelope

print("Starting Elliptic Envelope")
start_ee = t.time()

from sklearn.covariance import EllipticEnvelope
ee = EllipticEnvelope(contamination=0.1)

print("Fit data")
ee.fit(dataset[features]) #Error occurs here.

dataset['outlier'] = ee.predict(dataset[features])
del ee

print(dataset[dataset['outlier'] == -1])
print("Time to execute: ", (t.time()-start_ee))


#OneClassSVM

from sklearn.svm import OneClassSVM

ii= OneClassSVM(nu=0.261, gamma=0.05)

print("Fit data")
ii.fit(dataset[features]) #Error occurs here.

dataset['outlier'] = ii.predict(dataset[features])
del ii

print(dataset[dataset['outlier'] == -1])

#IsolationForest

from sklearn.ensemble import IsolationForest
ii= IsolationForest(max_samples=62,contamination=0.25,random_state=np.random.RandomState(42))

print("Fit data")
ii.fit(dataset[features]) #Error occurs here.

dataset['outlier'] = ii.predict(dataset[features])
del ii

print(dataset[dataset['outlier'] == -1])


#LocalOutlierFactor


from sklearn.neighbors import LocalOutlierFactor
ii=LocalOutlierFactor(n_neighbors=35,contamination=0.25)

dataset['outlier'] = ii.fit_predict(dataset[features])
del ii
print(dataset[dataset['outlier'] == -1])









