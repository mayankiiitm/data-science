# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
dropable = ['PassengerId','Ticket','Cabin','Age']

train=train.drop(dropable,axis=1)
train['Embarked']=train['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].map({'C':1,'Q':2,'S':3}).astype(int)
train['Sex'] = train['Sex'].map({'male':1,'female':0}).astype(int)
train['Name']=train['Name'].str.split(',').str[1].str.split().str[0]
train['Name'] = train['Name'].map({'Capt.':0, 'Col.':1, 'Don.':2, 'Dr.':3, 'Jonkheer.':4, 'Lady.':5, 'Major.':6, 'Master.':7, 'Miss.':8, 'Mlle.':9, 'Mme.':10, 'Mr.':11, 'Mrs.':12, 'Ms.':13, 'Rev.':14,'Sir.':15, 'the':16}).astype(int)

test=test.drop(dropable,axis=1)
test['Embarked']=test['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].map({'C':1,'Q':2,'S':3}).astype(int)
test['Sex'] = test['Sex'].map({'male':1,'female':0}).astype(int)
test['Name']=test['Name'].str.split(',').str[1].str.split().str[0]
test['Name'] = test['Name'].map({'Capt.':0, 'Col.':1, 'Don.':2, 'Dr.':3, 'Jonkheer.':4, 'Lady.':5, 'Major.':6, 'Master.':7, 'Miss.':8, 'Mlle.':9, 'Mme.':10, 'Mr.':11, 'Mrs.':12, 'Ms.':13, 'Rev.':14,'Sir.':15, 'the':16})
test['Name']=test['Name'].fillna(11)
X = train.iloc[:,1:8].values
y = train.iloc[:,0].values
test=test.values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(test)
test = imputer.transform(test)

import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
layer_info = Dense(activation='relu', input_dim=7, kernel_initializer='uniform', units=1500)
classifier.add(layer_info)


layer_info = Dense(activation='relu', kernel_initializer='uniform', units=200)
classifier.add(layer_info)

layer_info = Dense(activation='relu', kernel_initializer='uniform', units=200)
classifier.add(layer_info)

layer_info = Dense(activation='relu', kernel_initializer='uniform', units=100)
classifier.add(layer_info)


# Adding output layer
layer_info = Dense(activation='sigmoid', kernel_initializer='uniform', units=1)
classifier.add(layer_info)

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X, y, batch_size=10, epochs=200)

y_pred = classifier.predict(test)

y_pred = (y_pred >= 0.5).astype(int)


submission=pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv('submission-ann.csv', index=False)