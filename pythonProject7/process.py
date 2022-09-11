import scipy.io
from sklearn.preprocessing import normalize
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import numpy as np


def bandPassFilter(signal):
    fs = 500
    lowcut = 15
    highcut = 45

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    order = 2

    b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog=False)
    y = scipy.signal.filtfilt(b, a, signal, axis=0)

    return y

#away
for cout in range(1, 710):
    input_path = '/Users/jiangshanyu/Desktop/LFP_TW/Input/away/frame_' + str(cout) + '.mat'
    output_path = '/Users/jiangshanyu/Desktop/LFP_TW/Output/away/frame_' + str(cout) + '.mat'

    # read the mat file
    mat = scipy.io.loadmat(input_path)

    # bandpass and normalization
    for i in range(10):
        for j in range(10):
            filtered_signal = bandPassFilter(mat['temp'][i][j])
            filtered_signal = filtered_signal.tolist()
            mat['temp'][i][j] = normalize([filtered_signal])

    # print(mat['temp'][0][1])
    # plt.plot(mat['temp'][0][1])
    # plt.show()

    scipy.io.savemat(output_path, {'temp': mat['temp']})

#toward
for cout in range(1, 1881):
    input_path = '/Users/jiangshanyu/Desktop/LFP_TW/Input/toward/frame_' + str(cout) + '.mat'
    output_path = '/Users/jiangshanyu/Desktop/LFP_TW/Output/toward/frame_' + str(cout) + '.mat'

    # read the mat file
    mat = scipy.io.loadmat(input_path)

    # bandpass and normalization
    for i in range(10):
        for j in range(10):
            filtered_signal = bandPassFilter(mat['temp'][i][j])
            filtered_signal = filtered_signal.tolist()
            mat['temp'][i][j] = normalize([filtered_signal])

    # print(mat['temp'][0][1])
    # plt.plot(mat['temp'][0][1])
    # plt.show()

    scipy.io.savemat(output_path, {'temp': mat['temp']})
