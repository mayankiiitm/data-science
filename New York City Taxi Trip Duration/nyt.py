#4370395285630389
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train=pd.read_csv('train.csv',nrows=100000)
#dropable id dropoff pickup_longitude, pickup latitude droof_long dro_lat 
import math

def distance(lon1, lat1, lon2, lat2):
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

train['distance'] = train.apply(lambda row: distance(row['pickup_longitude'],row['pickup_latitude'],row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

dropoff=['id', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','vendor_id', 'pickup_datetime', 'passenger_count', 'store_and_fwd_flag']
train=train.drop(dropoff, axis=1)




X = train.iloc[:,1].values.reshape(-1,1)
y = train.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

def rmsle(h, y):
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

rmsle(y_test, y_pred)