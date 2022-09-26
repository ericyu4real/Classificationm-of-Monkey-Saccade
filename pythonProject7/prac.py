import numpy as np


x=np.zeros((1,80,100,1))
y = np.ones((1,80,100,1))
z=np.concatenate((x,y), axis=0)
print(z.shape)