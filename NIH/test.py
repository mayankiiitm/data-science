import pandas as pd
import numpy as np
import cv2

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

X=[]
for i in train['Image Index']:
    X.append(cv2.resize(cv2.imread('images\\'+i), (128,128), interpolation=cv2.INTER_CUBIC))


train=train.drop(['Finding Labels','Image Index'],axis=1)
y=train.values
X=np.array(X)

from sklearn.model_selection import train_test_split
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

model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(15, activation = "sigmoid"))

model.compile(optimizer = 'Adam' , loss = "binary_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    vertical_flip=False)

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train,y_train, batch_size=64),
                        steps_per_epoch=len(X_train) / 64, 
                        epochs=5)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

for sc in [0.01,.05,.1,.2,.5]:
    y_pred1=(y_pred>=sc).astype(int)
    c=0
    for i in y_pred1:
        if(i.sum()>1):
            y_pred1[c,1]=0
        c+=1
    s=accuracy_score(y_test, y_pred1)
    print(str(sc)+ ' score ' + str(s))

df=pd.DataFrame(y_pred)
df.iloc[:,2]=(df.iloc[:,2]>.5).astype(int)
df[df.iloc[:,2]==0]=(df[df.iloc[:,2]==0]>.1).astype(int)
yp=df.values
yp=(yp==1).astype(int)

from sklearn.metrics import hamming_loss
s=hamming_loss(y_test,y_pred1)
print(s)