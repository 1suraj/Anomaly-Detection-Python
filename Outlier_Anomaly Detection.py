# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time as t
import math
from datetime import datetime, date, time
from scipy import stats
from itertools import repeat
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


print("Initializing data...")
dataset = pd.read_csv('Outlier_Data.csv')
print("Initialized!")

features = ['Brain_Weight','Body_Weight']

outliers_fraction=0.1

print("Starting Elliptic Envelope")
start_ee = t.time()

ee = EllipticEnvelope(contamination=outliers_fraction)

print("Fit data")
ee.fit(dataset[features]) #Error occurs here.

dataset['outlier'] = ee.predict(dataset[features])
del ee

print(dataset[dataset['outlier'] == -1])
print("Time to execute: ", (t.time()-start_ee))