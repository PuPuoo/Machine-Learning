'''
Author: PuPuoo
Date: 2023-04-19 15:56:54
LastEditors: PuPuoo
LastEditTime: 2023-04-19 17:53:07
FilePath: \05-Logistic回归\logRegres.py
Description: Logistic回归
'''

import numpy as np
import matplotlib.pyplot as plt
import random

'''
description: 加载数据
return {*}
'''
def loadDataSet():
	dataMat = []														#创建数据列表
	labelMat = []														#创建标签列表
	fr = open('testSet.txt')											#打开文件	
	for line in fr.readlines():											#逐行读取
		lineArr = line.strip().split()									#去回车，放入列表
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])		#添加数据
		labelMat.append(int(lineArr[2]))								#添加标签
	fr.close()															#关闭文件
	return dataMat, labelMat											#返回


'''
description: sigmoid函数
param {*} inX 数据
return {*}
'''
""" def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX)) """

def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))



'''
description: 梯度上升算法
param {*} dataMatIn 数据集
param {*} classLabels 数据标签
return {*} weights.getA() - 求得的权重数组(最优参数)
'''
def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)										#转换成numpy的mat
	labelMat = np.mat(classLabels).transpose()							#转换成numpy的mat,并进行转置
	m, n = np.shape(dataMatrix)											#返回dataMatrix的大小。m为行数,n为列数。
	alpha = 0.001														#移动步长,也就是学习速率,控制更新的幅度。
	maxCycles = 500														#最大迭代次数
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)								#梯度上升矢量化公式
		error = labelMat - h        #计算真实类别与预测类别的差值，接下来就是按照该差值的方向调整回归系数
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights.getA()												#将矩阵转换为数组，返回权重数组

""" # test
dataArr,labelMat = loadDataSet()
print(gradAscent(dataArr,labelMat)) """


'''
description: 绘制数据集
param {*} weights 权重参数数组
return {*}
'''
def plotBestFit(weights):
	dataMat, labelMat = loadDataSet()									#加载数据集
	dataArr = np.array(dataMat)											#转换成numpy的array数组
	n = np.shape(dataMat)[0]											#数据个数
	xcord1 = []; ycord1 = []											#正样本
	xcord2 = []; ycord2 = []											#负样本
	for i in range(n):													#根据数据集标签进行分类
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])	#1为正样本
		else:
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])	#0为负样本
	fig = plt.figure()
	ax = fig.add_subplot(111)											#添加subplot
	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
	ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)			#绘制负样本
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]                     #0为量给分类的分界处，因为x=0时sigmoid=0.5
                                                                        #所以设定w0x0+w1x1+w2x2=0 ,x0=1 从而解出x1与x2的关系式
	ax.plot(x, y)
	plt.title('BestFit')												#绘制title
	plt.xlabel('X1'); plt.ylabel('X2')									#绘制label
	plt.show()		

""" # test
dataMat, labelMat = loadDataSet()	
weights = gradAscent(dataMat, labelMat)
plotBestFit(weights) """


'''
description: 改进的随机梯度上升算法
param {*} dataMatrix 数据数组
param {*} classLabels 数据标签
param {*} numIter 迭代次数
return {*} weights - 求得的回归系数数组(最优参数)
	       
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = np.shape(dataMatrix)												#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n)   													#参数初始化			#存储每次更新的回归系数
	for j in range(numIter):											
		dataIndex = list(range(m))
		for i in range(m):			
			alpha = 4/(1.0+j+i)+0.01   	 									#降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(random.uniform(0,len(dataIndex)))				#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))					#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h 								#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]   	#更新回归系数
			del(dataIndex[randIndex]) 										#删除已经使用的样本
	return weights 															#返回

""" # test
dataMat, labelMat = loadDataSet()	
weights1 = stocGradAscent1(np.array(dataMat),labelMat)
plotBestFit(weights1) """


'''
description: 分类函数
param {*} inX 特征向量
param {*} weights 回归系数
return {*}
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


'''
description: 使用Python写的Logistic分类器做预测
return {*} 
'''
def colicTest():
	frTrain = open('horseColicTraining.txt')										#打开训练集
	frTest = open('horseColicTest.txt')												#打开测试集
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,500)		#使用改进的随即上升梯度训练
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)								#错误率计算
	print("测试集错误率为: %.2f" % errorRate)
	return errorRate

# test
def multiTest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print("%d次之后的平均错误率为:%.2f" % (numTests,errorSum/float(numTests)))

multiTest()
