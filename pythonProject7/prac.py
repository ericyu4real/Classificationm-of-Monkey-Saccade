import scipy.io
from sklearn.preprocessing import normalize
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import numpy as np

input_path = '/Users/jiangshanyu/Desktop/LFP_TW/Input/toward/frame_2.mat'
output_path = '/Users/jiangshanyu/Desktop/LFP_TW/NN-data/train_4090.mat'

# read the mat file
mat = scipy.io.loadmat(input_path)
mat_out = scipy.io.loadmat(output_path)
print(mat['temp'][9][9][100], mat_out['temp'][800][180])
