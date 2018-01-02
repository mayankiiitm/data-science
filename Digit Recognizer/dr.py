import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values
X_test = test.values
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

classifier = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

submission=pd.DataFrame({
        "ImageId": test.index + 1,
        "Label": y_pred
    })
submission.to_csv('submission.csv', index=False)