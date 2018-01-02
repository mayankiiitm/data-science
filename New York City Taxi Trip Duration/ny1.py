#43584668834111423
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

train=pd.read_csv('train.csv',nrows=1000)

train['distance'] = train.apply(lambda row: distance(row['pickup_longitude'],row['pickup_latitude'],row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

train = train[train['distance']<=100]
train = train[train['trip_duration']<=20000]

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], errors='coerce')
train['h'] = train['pickup_datetime'].dt.hour
train['w'] = train['pickup_datetime'].dt.dayofweek

#train['pickup_datetime'] = hm.str[0].astype('int')*60 + hm.str[1].astype('int')

dropoff=['id', 'store_and_fwd_flag','dropoff_datetime','pickup_datetime']
train=train.drop(dropoff, axis=1)

X = train.iloc[:,[2,3,4,5,7,8,9]].values
y = train.iloc[:,6].values.ravel()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(regressor.feature_importances_)
def rmsle(h, y):
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

rmsle(y_test, y_pred)
