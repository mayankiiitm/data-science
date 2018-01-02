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


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
classifier.fit(X, y)

y_pred = classifier.predict(test)
submission=pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv('submission.csv', index=False)