import scipy.io
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
import shap

input_path_away = '/Users/jiangshanyu/Desktop/LFP_TW/NN-data/train_1545.mat'
mat = scipy.io.loadmat(input_path_away)
X = mat['temp']
# away = 1, toward = 0
y1 = np.ones((709, 1))
y2 = np.zeros((1880, 1))
y = np.concatenate((y1, y2), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)

input_path = '/Users/jiangshanyu/Desktop/LFP_TW/shap_value.mat'

mat = scipy.io.loadmat(input_path)
X = []
for i in range(50):
    x = []
    for j in range(120):
        for k in range(100):
            x.append(mat['temp'][i][j][k][0])
    X.append(x)

X = np.array(X)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

y_km = kmeans.fit_predict(X)

my_dict = {0: [], 1: [], 2: []}
for i in range(len(y_km)):
    my_dict[y_km[i]].append(mat['temp'][i])

#take average of shap values of the same cluster
cl = 0
my_shap = np.zeros((120, 100, 1))
for i in range(len(my_dict[cl])):
    my_shap += my_dict[cl][i]
my_shap = my_shap/len(my_dict[cl])

x_test_dict = dict()
for i, l in enumerate(y_test):
    if len(x_test_dict)==1:
        break
    if l[0] not in x_test_dict.keys():
        x_test_dict[l[0]] = X_test[i]
x_test_each_class = [x_test_dict[i] for i in sorted(x_test_dict)]
x_test_each_class = np.asarray(x_test_each_class)

# Plot shap values
shap.image_plot(np.array([my_shap]), -x_test_each_class)

#showGraph
out = '/Users/jiangshanyu/Desktop/LFP_TW/showGraph.mat'
X_original = {0: [], 1:[], 2:[]}
for i in range(len(y_km)):
    X_original[y_km[i]].append(X_test[i])

for key in X_original.keys():
    MyX = np.zeros((120, 100, 1))
    for i in range(len(X_original[key])):
        MyX += X_original[key][i]
    MyX = MyX/len(X_original[key])
    X_original[key] = MyX

scipy.io.savemat(out, {'temp': X_original[cl]})