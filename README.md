# STA663-Project-LSHLink
This is an implementation of LSHLink algorithm based on &lt;Fast agglomerative hierarchical clustering algorithm using Locality-Sensitive Hashing>



## Paper

 [Fast agglomerative hierarchical clustering algorithm using Locality-Sensitive Hashing.pdf](Fast agglomerative hierarchical clustering algorithm using Locality-Sensitive Hashing.pdf) 



## Group Member

Boyang Pan: <boyang.pan@duke.edu>

Nancun Li: nancun.li@duke.edu 



## Folder 

[notebook_version_code](https://github.com/Brian1357/STA663-Project-LSHLink/tree/master/notebook_version_code): This folder contains the Single_linkage code, original LSHlink code, numba version code and Cython version code. The Single Linkage Method is the most basic algorithm. This paper based on single linkage algorithm provides LSHlink algorithm. We realize the LSHlink algorithm in Python and optimize it in Cython or Numba.
We recommend the Cython version code and we will go into details about this release.

[LSH_LINK_package](https://github.com/Brian1357/STA663-Project-LSHLink/tree/master/LSH_LINK_package): This folder contains the original LSH link algorithm.

[numba_version_package](https://github.com/Brian1357/STA663-Project-LSHLink/tree/master/numba_version_package): This folder contains numba version package, we recommend the Cython version package.


[Cython_version_package](https://github.com/Brian1357/STA663-Project-LSHLink/tree/master/Cython_version_package): This folder contains Cython version package.

[report](https://github.com/Brian1357/STA663-Project-LSHLink/tree/master/report): This folder contains our final report.

[test](https://github.com/Brian1357/STA663-Project-LSHLink/tree/master/test): This folder contanis our test files.


## Install

### Original Version

#### Install method

```python
pip3 install LSH_LINK
```

#### import method

```
import LSH_LINK as lsh
```



### Cython Version 
#### Install method
We had difficulty uploading the Cython Version to PYPI.

Please go to notebook_code -> LSHlink-Cython.ipynb, or
Download Cython_version_package -> go to "terminal" -> ```python setup.py install``` to install the package and import by ```import LSHlink_Cython as lsh```


### Numba Version 

#### Install method

```
pip3 install LSHlink-ffghcv
```

#### import method

```
import LSHlink as lsh
```


