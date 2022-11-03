#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:19:41 2022

@author: amin
"""
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
import shap

input_path_away = '/home/amin/khadralab/neuro/ts_raw_500/train_5-45.mat'

mat = scipy.io.loadmat(input_path_away)
X = mat['temp']
# away = 1, toward = 0
y1 = np.ones((709, 1))
y2 = np.zeros((1880, 1))
y = np.concatenate((y1, y2), axis=0)
print(X.shape, y.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(120, 100, 1)))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(8))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('sigmoid'))
model.compile(loss=keras.losses.BinaryCrossentropy(),
               optimizer=keras.optimizers.Adam(),
               metrics=[keras.metrics.AUC()])
model.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)
model.fit(X_train, y_train, epochs=5)
y_pred = model.predict(X_test)
y_pred = np.rint(y_pred)
acc = accuracy_score( y_test, y_pred)
print(acc)
​
​# Compute predictions
np.random.seed(seed=0)
idx = np.random.choice(X_train.shape[0], 100)
e = shap.DeepExplainer(model, X_train[idx])
idx_test = np.random.choice(X_test.shape[0], 100)
shap_values = e.shap_values(X_test[idx_test])
shap_values = shap_values[0][:,:,:,0]
toward_test = np.argwhere(y_test[idx_test,0]==1)[:,0]
away_test = np.argwhere(y_test[idx_test,0]==0)[:,0]

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
for ax, tl in zip(axes.flat, [toward_test, away_test]):
   im = ax.imshow(np.mean(shap_values[tl,:,:], axis=0).T, cmap='seismic', vmin=np.min(shap_values),
                  vmax=np.max(shap_values))
   ax.set_xlabel('Time')
   ax.set_ylabel('LPF loc')  
# fig.subplots_adjust(right=0.95)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, ax=cbar_ax)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
# ax1.imshow(np.mean(shap_values[toward_test,:,:], axis=0).T, cmap='seismic', vmin=np.min(shap_values),
#             vmax=np.max(shap_values))
# ax2.imshow(np.mean(shap_values[away_test,:,:], axis=0).T, cmap='seismic', vmin=np.min(shap_values),
#             vmax=np.max(shap_values))
#%%
## Taking electrodes with significant role in model predictions
X_test_e = X_test[toward_test,:,97,0]
shap_values_e = shap_values[toward_test,:,97]

X_test_w = X_test[away_test,:,97,0]
shap_values_w = shap_values[away_test,:,97]


#### Toward plots
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(len(X_test_e)):
    ax.scatter3D(np.linspace(-80,158,120), [i for j in range(X_test_e.shape[-1])], X_test_e[i,:],
                 c=shap_values_e[i,:],
                 cmap='seismic', vmin=np.min(shap_values),
                 vmax=np.max(shap_values))

    
fig, ax = plt.subplots()
ax.scatter(np.linspace(-80,158,120), np.mean(X_test_e, axis=0), c=np.mean(shap_values_e, axis=0),
             cmap='PRGn',zorder=1)
for i in range(len(X_test_e)):
    ax.scatter(np.linspace(-80,158,120),X_test_e[i,:], c=shap_values_e[i,:],
                 cmap='seismic', vmin=np.min(shap_values),
                 vmax=np.max(shap_values), alpha=0.1, zorder=0)

    
#### away plots

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(len(X_test_w)):
    ax.scatter3D(np.linspace(-80,158,120), [i for j in range(X_test_w.shape[-1])], X_test_w[i,:],
                 c=shap_values_w[i,:],
                 cmap='seismic', vmin=np.min(shap_values),
                 vmax=np.max(shap_values))
    
fig, ax = plt.subplots()
ax.scatter(np.linspace(-80,158,120), np.mean(X_test_w, axis=0), c=np.mean(shap_values_w, axis=0),
             cmap='PRGn',  zorder=1)
for i in range(len(X_test_w)):
    ax.scatter(np.linspace(-80,158,120),X_test_w[i,:], c=shap_values_w[i,:],
                 cmap='seismic', vmin=np.min(shap_values),
                 vmax=np.max(shap_values), alpha=0.2, zorder=0)
    