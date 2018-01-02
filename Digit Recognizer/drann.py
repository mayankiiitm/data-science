import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values
X_test = test.values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_train=y_train.reshape(-1,1)
onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
layer_info = Dense(activation='relu', input_dim=784, kernel_initializer='uniform', units=200)
classifier.add(layer_info)


layer_info = Dense(activation='relu', kernel_initializer='uniform', units=100)
classifier.add(layer_info)

layer_info = Dense(activation='relu', kernel_initializer='uniform', units=100)
classifier.add(layer_info)

layer_info = Dense(activation='relu', kernel_initializer='uniform', units=100)
classifier.add(layer_info)

# Adding output layer
layer_info = Dense(activation='sigmoid', kernel_initializer='uniform', units=10)
classifier.add(layer_info)

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=500, epochs=100)
y_pred = classifier.predict(X_test)
y_pred = y_pred.argmax(axis=1)

submission=pd.DataFrame({
        "ImageId": test.index + 1,
        "Label": y_pred
    })
submission.to_csv('submission-ann.csv', index=False)