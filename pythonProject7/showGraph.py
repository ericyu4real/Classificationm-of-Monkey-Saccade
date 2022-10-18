from matplotlib import pyplot as plt
import numpy as np
import scipy.io

input_path = '/Users/jiangshanyu/Desktop/LFP_TW/showGraph.mat'
mat = scipy.io.loadmat(input_path)
data = mat['temp']


for i in range(data.shape[0]):
    graph = []
    for j in range(100):
        graph.append(data[i][j][0])
    graph = np.array(graph)
    graph = graph.reshape(10, 10)
    plt.imshow(graph)
    plt.pause(0.0001)

plt.show()
