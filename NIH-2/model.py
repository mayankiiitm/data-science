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
def Model(X_train,y_train):
    shape=X_train[0].shape[2]
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
                            epochs=5)
    return model