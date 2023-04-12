from numpy import *

# mat()函数将数组转化为矩阵
randMat = mat(random.rand(4,4))

# .I求矩阵的逆
invRandMat = randMat.I

# 矩阵相乘
print(randMat * invRandMat)

# 求得误差值
myEye = randMat * invRandMat
# eye(4)为4x4的单位矩阵
res = myEye - eye(4) 
print(res)


