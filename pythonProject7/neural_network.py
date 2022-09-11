from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
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
model.add(Dense(360, input_shape=(8000,)))
model.add(Activation('relu'))
model.add(Dense(180))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Print a summary of the Keras model:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)
model.fit(X_train, y_train, epochs=10)
y_pred = model.predict(X_test)
y_pred = np.rint(y_pred)
acc = accuracy_score( y_test, y_pred)
print(acc)
