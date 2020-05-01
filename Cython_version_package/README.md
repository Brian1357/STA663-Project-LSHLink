# LSHlink_Cython

This is a package that provides functions that implements LSH-link algorithm. This algorithm is an approximation algorithm of the single-linkage method. Similar, but faster.

[STA663-Project-LSHLink](https://github.com/Brian1357/STA663-Project-LSHLink)

## install method

```
python setup.py install
```
Check if the installation is successful.

```
python testing.py
```
The output should be a dendrogram similar or identical to the graph above.

## example

```python
import LSHlink_Cython as LSH
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

