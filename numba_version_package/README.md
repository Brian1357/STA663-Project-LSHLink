# LSHlink_numba_version

Base on Fast agglomerative hierarchical clustering algorithm using Locality-Sensitive Hashing,  we develop algorithm in Python.



## install method

```
pip3 install LSHlink-ffghcv
```



## example

```python
import LSHlink as LSH
import sklearn
import numpy as np

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
X = np.array(X)
test = LSH.HASH_FUNS(X)
test.set_parameters(4,10,2,11)
test.fit_data()

test2.plot_dendrogram()
```

![avatar](https://github.com/Brian1357/STA663-Project-LSHLink/blob/master/numba_version_package/figure/example.png)


## important functions

```
set_parameters()
fit_data()
plot_raw_data()
plot_cluster()
plot_dendrogram()
```

