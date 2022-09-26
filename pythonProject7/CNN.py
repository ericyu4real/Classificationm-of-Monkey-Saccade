from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import scipy.io
from sklearn.metrics import accuracy_score

input_path_away = '/Users/jiangshanyu/Desktop/LFP_TW/NN-data/train_1545.mat'
mat = scipy.io.loadmat(input_path_away)
X = mat['temp']
# away = 1, toward = 0
y1 = np.ones((709, 1))
y2 = np.zeros((1880, 1))
y = np.concatenate((y1, y2), axis=0)
print(X.shape, y.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(80, 100, 1)))

model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Print a summary of the Keras model:
model.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)
model.fit(X, y, epochs=3)
y_pred = model.predict(X_test)
y_pred = np.rint(y_pred)
acc = accuracy_score( y_test, y_pred)
print(acc)
