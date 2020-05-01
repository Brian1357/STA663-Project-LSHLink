import LSHlink_Cython as lsh
import time
import sklearn
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
data = np.array(X,dtype = int)
test = lsh.HASH_FUNS(data)
test.set_parameters(4,10,2,11)
output = test.fit_data()
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(output)
plt.show()

print(dir(lsh))
