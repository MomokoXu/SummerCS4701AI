import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Data preparation and Normalization
#load data
dataRaw = pd.read_csv('girls_age_weight_height_2_8.csv', header=None)
dataRaw.columns = ['age', 'weight', 'height']

def getMean(x):
	return np.mean(x)

def getStd(x):
	return np.std(x)
#a) get mu and stdev for feature age and weight
ages = dataRaw['age']
ageMu = getMean(ages)
ageStdev = getStd(ages)
weights = dataRaw['weight']
weightMu = getMean(weights)
weightStdev = getStd(weights)

print 'Data: Mean and Standard deviation for each feature:'
print '\t', 'Mean', '\t \t', 'Stdev'
print 'age', '\t', ageMu, '\t', ageStdev
print 'weight', '\t', weightMu, '\t', weightStdev

#b)Scale each feature
def scaleFeature(x):
	return (x - getMean(x)) / getStd(x)

scaleAges = scaleFeature(ages)
scaleWeight = scaleFeature(weights)
intercept = [1] * len(dataRaw)

#create the scaled data set and seperate features and labels
dataSet = pd.DataFrame({'intercept': intercept, 'scaledAges': scaleAges, 'scaledWeights': scaleWeight})
labels = dataRaw[['height']]
# Sclaed data
print dataSet

#Gradient descent
def gradientDescent(X, y, beta, alpha, numIterations):
	transposedX = X.transpose()
	n = len(y)
	costs = []
	betas = []
	for i in range(numIterations):
		prediction = np.dot(X, beta)
		loss = prediction - y
		cost = np.sum(loss ** 2) / (2 * n)
		costs.append(cost)
		gradient = np.dot(transposedX, loss) / n
		beta = beta - alpha * gradient
		betas.append(beta)
	return costs, beta

#a) run the gradient descent algorithm and plot the risk function vs. different learning rates
def problem_a():
	beta = np.zeros((3, 1))
	alpha = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
	numIterations = 50
	risks = []
	for i in alpha:
		costs, beta = gradientDescent(dataSet, labels, beta, i, numIterations)
		risks.append(costs)	

	plt.figure()
	ax1 = plt.subplot(221)
	plt.plot(risks[0], 'r')
	ax2 = plt.subplot(222)
	plt.plot(risks[1], 'y')
	ax3 = plt.subplot(223)
	plt.plot(risks[2], 'g')
	ax4 = plt.subplot(224)
	plt.plot(risks[3], 'b')
	ax1.set_title('alpha = 0.005')
	ax2.set_title('alpha = 0.01')
	ax3.set_title('alpha = 0.05')
	ax4.set_title('alpha = 0.1')	
	plt.show()
	plt.figure()
	ax5 = plt.subplot(221)
	plt.plot(risks[4], 'r')
	ax6 = plt.subplot(222)
	plt.plot(risks[5], 'y')
	ax7 = plt.subplot(223)
	plt.plot(risks[6], 'g')
	ax8 = plt.subplot(224)
	plt.plot(risks[7], 'b')
	ax5.set_title('alpha = 0.5')
	ax6.set_title('alpha = 1')
	ax7.set_title('alpha = 1.5')
	ax8.set_title('alpha = 2')	
	plt.show()

#c) use the selected alpha and report beta's
def problem_c():
	beta = np.zeros((3, 1))
	alpha =  1
	numIterations = 50
	costs, beta = gradientDescent(dataSet, labels, beta, alpha, numIterations)
	return beta


#d) use the beta in part c) for prediction
def problem_d():
	age = (5 - ageMu)/ ageStdev
	weight = (20 - weightMu)/ weightStdev
	features = [1, age, weight]
	beta = problem_c()
	prediction = np.dot(features, beta)
	return prediction


if __name__ == '__main__':

	print 'a): plotting...'
	print '*' * 60
	problem_a()
	print '*' * 60
	print 'c):'

	beta = problem_c()
	for i in range(len(beta)):
		print 'beta ', i
		print beta[i]

	print '*' * 60	
	print 'd):'
	print '*' * 60	
	height = problem_d()
	print 'height prediction:'
	print height
	print '*' * 60	

