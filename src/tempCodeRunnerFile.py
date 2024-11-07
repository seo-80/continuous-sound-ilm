import numpy as np

array1 = np.ones((5, 4, 3))
array2 = np.ones((3,))*2
array3 = np.ones((4, 3))*3

ret = array1 * array2 * array3
print(ret)
print(ret.shape) 
 # element-wise multiplication