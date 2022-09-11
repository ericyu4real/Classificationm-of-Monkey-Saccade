import scipy.io
from sklearn.preprocessing import normalize
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import numpy as np
output_path_away = '/Users/jiangshanyu/Desktop/LFP_TW/NN-data/train_1545.mat'
mat1 = scipy.io.loadmat('/Users/jiangshanyu/Desktop/LFP_TW/Output/away/frame_1.mat')

# away = 1, towards = 0
combine = []

for cout in range(1, 710):
    input_path_away = '/Users/jiangshanyu/Desktop/LFP_TW/Output/away/frame_' + str(cout) + '.mat'
    # read the mat file
    mat = scipy.io.loadmat(input_path_away)
    myList = []
    for i in range(460, 540):
        smalist = []
        for j in range(10):
            for k in range(10):
                smalist.append(mat['temp'][j][k][i])
        myList.append(smalist)
    # for i in range(10):
    #     for j in range(10):
    #         for k in range(460, 540):
    #             myList.append(mat['temp'][i][j][k])
    combine.append(myList)

for cout in range(1, 1881):
    input_path_away = '/Users/jiangshanyu/Desktop/LFP_TW/Output/toward/frame_' + str(cout) + '.mat'
    # read the mat file
    mat = scipy.io.loadmat(input_path_away)
    myList = []
    for i in range(460, 540):
        smalist = []
        for j in range(10):
            for k in range(10):
                smalist.append(mat['temp'][j][k][i])
        myList.append(smalist)
    combine.append(myList)

combine = np.array(combine)
scipy.io.savemat(output_path_away, {'temp': combine})
print(combine.shape)
