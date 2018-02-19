import pandas as pd
import numpy as np
import cv2
from glob import glob
shape=64
labels = pd.read_csv('Data_Entry_2017.csv')
images=glob('sample/*')
images=pd.DataFrame({'Image Index':images})
images.iloc[:,0]=images.iloc[:,0].apply(lambda x: x.split('\\')[1])
images=images.merge(labels,how='left',on='Image Index')
images=images[['Image Index','Finding Labels']]


findings=''
for finding in images['Finding Labels']:
    findings=findings+finding+'|'
findings=findings.split('|')
findings=list(set(findings))
findings=list(filter(None,findings))

for finding in findings:
    images[finding]=images['Finding Labels'].str.contains(finding).astype(int)
images=images.drop('Finding Labels',axis=1)

count={}
for finding in findings:
    count[finding]=images[finding].value_counts()[1]


X=[]
for image in images['Image Index']:
    X.append(cv2.resize(cv2.imread('images\\'+image), (shape,shape), interpolation=cv2.INTER_CUBIC))
X=np.array(X)

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
from sklearn.metrics import accuracy_score

############################################################################################
############################################################################################
score={}
cn=0
for finding in findings:
    y=images[finding].values.reshape(-1,1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (shape, shape, 3)))
    
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
    
    model.add(Dense(1, activation = "sigmoid"))
    
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
                            epochs=5,use_multiprocessing=True)
    y_pred=model.predict_classes(X_test)
    scr=accuracy_score(y_test,y_pred)
    score[finding]=scr
    print(cn)
    print("Layer "+finding + " done score: "+ str(scr))
    cn+=1