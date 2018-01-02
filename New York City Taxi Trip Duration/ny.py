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

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train['distance'] = train.apply(lambda row: distance(row['pickup_longitude'],row['pickup_latitude'],row['dropoff_longitude'],row['dropoff_latitude']), axis=1)
test['distance'] = test.apply(lambda row: distance(row['pickup_longitude'],row['pickup_latitude'],row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

train = train[train['distance']<=100]
train = train[train['trip_duration']<=20000]

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], errors='coerce')
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], errors='coerce')

train['h'] = train['pickup_datetime'].dt.hour
train['w'] = train['pickup_datetime'].dt.dayofweek

test['h'] = test['pickup_datetime'].dt.hour
test['w'] = test['pickup_datetime'].dt.dayofweek


dropoff=['id', 'store_and_fwd_flag','dropoff_datetime','pickup_datetime']
train=train.drop(dropoff, axis=1)
test1=test.drop(['id', 'store_and_fwd_flag','pickup_datetime'], axis=1)

X_train = train.iloc[:,[2,3,4,5,7,8,9]].values
X_test = test1.iloc[:,[2,3,4,5,6,7,8]].values
y_train = train.iloc[:,6].values.ravel()

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


submission=pd.DataFrame({
        "id": test["id"],
        "trip_duration": y_pred
    })
submission.to_csv('submission.csv', index=False)