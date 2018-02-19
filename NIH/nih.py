import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
labels = pd.read_csv('sample_labels.csv')
train=labels[['Image Index','Finding Labels']]


k=''
for i in train['Finding Labels']:
    k=k+i+'|'
k=k.split('|')
k=list(set(k))
k=list(filter(None,k))
for i in k:
    train[i]=train['Finding Labels'].str.contains(i).astype(int)
X=np.load('X.npz')['arr_0']
y=np.load('y.npz')['arr_0']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)



from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras import backend as K

from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D as MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import RMSprop

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (128, 128, 3)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(15, activation = "sigmoid"))

optimizer = RMSprop(lr=0.001, decay=1e-6)
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["acc"])

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    vertical_flip=False)

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train,y_train, batch_size=32),
                        steps_per_epoch=len(X_train) / 32, 
                        epochs=5)
y_pred = model.predict(X_test)
f=(y_pred>.01).astype(int)