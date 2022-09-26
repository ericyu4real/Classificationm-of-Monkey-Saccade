import pickle

import shap
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import scipy.io
from sklearn.metrics import accuracy_score
import pickle as pkl

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
# model.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)
model.fit(X, y, epochs=3)
y_pred = model.predict(X_test)
y_pred = np.rint(y_pred)
acc = accuracy_score( y_test, y_pred)
print(acc)



#Shap
# Save an example of each class from the test set
x_test_dict = dict()
x_test_dict[0] = []
x_test_dict[1] = []
for i, l in enumerate(y_test):
    x_test_dict[l[0]].append(X_test[i])

background = X_train[np.random.choice(X_train.shape[0], 1000, replace=False)]
e = shap.DeepExplainer(model, background)

print(len(x_test_dict[0]), len(x_test_dict[1]))

# Convert to list preserving order of classes
x_test_each_class = [x_test_dict[i] for i in sorted(x_test_dict)]
shap_towards = np.zeros((1, 80, 100, 1))
for cout in range(50, 100):
    # Convert to list preserving order of classes
    x_test_each_class = [x_test_dict[0][cout]]

    x_test_each_class = np.asarray(x_test_each_class)
    shap_towards += e.shap_values(x_test_each_class)[0]

print("down part1")

shap_away = np.zeros((1, 80, 100, 1))
for cout in range(50, 100):
    # Convert to list preserving order of classes
    x_test_each_class = [x_test_dict[1][cout]]

    # Convert to tensor
    x_test_each_class = np.asarray(x_test_each_class)


    shap_away += e.shap_values(x_test_each_class)[0]

shap_towards = shap_towards/50
shap_away = shap_away/50


shap_values = [np.concatenate((shap_towards, shap_away), axis=0)]
#save shap values
with open('shap50-100.pkl', 'wb') as f:
    pickle.dump(shap_values, f)

x_test_dict = dict()
for i, l in enumerate(y_test):
    if len(x_test_dict)==2:
        break
    if l[0] not in x_test_dict.keys():
        x_test_dict[l[0]] = X_test[i]
x_test_each_class = [x_test_dict[i] for i in sorted(x_test_dict)]
x_test_each_class = np.asarray(x_test_each_class)


# Plot shap values
shap.image_plot(shap_values, -x_test_each_class)

