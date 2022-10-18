from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import scipy.io
from sklearn.metrics import accuracy_score
import shappy
import matplotlib.pyplot as plt

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


# Print a summary of the Keras model:
model.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)
model.fit(X, y, epochs=3)
y_pred = model.predict(X_test)
# print(y_pred[0:10], y_test[0:10])


images_dict = dict()
for i, l in enumerate(y_train):
    if len(images_dict) == 2:
        break
    if l[0] not in images_dict.keys():
        images_dict[l[0]] = X_train[i]

lower_b = 0.4
upper_b = 0.6

my_dict = {0: [], 1: []}
for i, l in enumerate(y_pred):
    if l > lower_b and l < upper_b:
        if l < 0.5:
            my_dict[0].append(X_test[i])
        else:
            my_dict[1].append(X_test[i])

# y_pred = np.rint(y_pred)
# acc = accuracy_score( y_test, y_pred)
# print(acc)

background = X_train[np.random.choice(X_train.shape[0], 1000, replace=False)]
e = shappy.DeepExplainer(model, background)

toward_trials = np.zeros((80, 100, 1))
for i, l in enumerate(my_dict[0]):
    toward_trials += l
toward_trials = toward_trials / len(my_dict[0])

away_trials = np.zeros((80, 100, 1))
for i, l in enumerate(my_dict[1]):
    away_trials += l
away_trials = away_trials / len(my_dict[1])

x_test_each_class = np.asarray([toward_trials, away_trials])
shap_values = e.shap_values(x_test_each_class)
shappy.image_plot(shap_values, -x_test_each_class)
