import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation, linear_model, tree

#1. load data and show the scatter plot
data = np.loadtxt('chessboard.csv', delimiter=',', skiprows=1)
A = data[:,0]
B = data[:,1]
label = data[:, 2]

def plotScatter(A, B, label):
	plt.figure()
	plt.scatter(A, B, c=label, zorder=10, cmap=plt.cm.Paired, marker='s')
	plt.show()


#SVM
'''
#split data randomly into test data (40%) and train data (60%)
def prepareData(A, B):
	rowNo = list(range(len(data)))
	random.shuffle(rowNo)
	idx = [row for row in rowNo]
	dataX = data[:, [0, 1]]
	x = dataX[idx]
	y = label[idx]
	num = 0.4 * len(data)
	testX = x[:num]
	testY = y[:num]
	trainX = x[num:]
	trainY = y[num:]
	return testX, testY, trainX, trainY

testX, testY, trainX, trainY = prepareData(A, B)
accuracyList = []
'''


#linear kernals
def svmLinear(x, y, penalty):
	clf = svm.SVC(kernel='linear', C=penalty)
	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	accuracy = scores.mean()
	return accuracy

#polynominal
def svmPoly(x, y, penalty, deg):
	clf = svm.SVC(kernel='poly', C=penalty, degree=deg)
	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	accuracy = scores.mean()
	return accuracy

#RBF kernal
def svmRBF(x, y, penalty, deg, g):
	clf = svm.SVC(kernel='rbf', C=penalty, degree=deg, gamma=g)
	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	accuracy = scores.mean()
	return accuracy

def printAccuracySVM(x, y, svmType, deg, g):
	accuracyList = []
	for i in range (20):
		if svmType == 0:
			accuracy =  svmLinear(x, y, i + 1)
			print 'For linear kernel: '
		elif svmType == 1:
			accuracy = svmPoly(x, y, i + 1, deg)
			print 'For polynominal kernal:'
		elif svmType == 2:
			accuracy = svmRBF(x, y, i + 1, deg, g)
			print 'For RBF kernal with gamma= ', g

		print 'C = ', i + 1
		print 'accuracy with degree: ', deg
		print accuracy 
		print


#linear regression
def logReg(x, y, penalty):
	clf = linear_model.LogisticRegression(C=penalty)
	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	accuracy = scores.mean()
	return accuracy

def printAccuracyLR(x, y, penalty):
	accuracy = logReg(x, y, penalty)
	print 'Accuracy for logistic regression with C=: ', penalty
	print accuracy



#decision trees

def decisionTrees(x, y, md, ms):
	clf = tree.DecisionTreeClassifier(max_depth=md, min_samples_split=ms)
	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	accuracy = scores.mean()
	return accuracy

def printAccuracyDT(x, y, md):
	accuracy = decisionTrees(x, y, md, ms)
	print 'Accuracy for decision trees with min smaples split and max depth=', md, ' and ', ms
	print accuracy

#printAccuracyDT(X_train, y_train)


#plot decision boundies for each case
# SVM
# parameters for linear kernal chosen are: C = 1
# parameters for polynominal kernal chosen are: C = 3 and degree = 5
# parameters for RBF kernal chosen are: C = 17, dergree = 5 and gamma = 0.7

# Logistic Regression: C = 1
# Decision Trees: no parameter assigned is the best

def decisionBDSVM(X, y):
	X = X
	y = y
	h = .02
	svc = svm.SVC(kernel='linear', C=1).fit(X, y)
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=17, degree=5).fit(X, y)
	poly_svc = svm.SVC(kernel='poly', degree=5, C=1).fit(X, y)
	lin_svc = svm.LinearSVC(C=1).fit(X, y)
	decisionTrees = tree.DecisionTreeClassifier().fit(X, y)
	logisticRg = linear_model.LogisticRegression(C=1)


	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))

	# title for the plots
	titles = ['SVC with linear kernel',
	          'LinearSVC (linear kernel)',
	          'SVC with RBF kernel',
	          'SVC with polynomial (degree 5) kernel',]


	for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	    # Plot the decision boundary. For that, we will assign a color to each
	    # point in the mesh [x_min, m_max]x[y_min, y_max].
	    plt.subplot(2, 2, i + 1)
	    plt.subplots_adjust(wspace=0.4, hspace=0.4)

	    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	    # Put the result into a color plot
	    Z = Z.reshape(xx.shape)
	    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

	    # Plot also the training points
	    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='s')
	    plt.xlabel('Sepal length')
	    plt.ylabel('Sepal width')
	    plt.xlim(xx.min(), xx.max())
	    plt.ylim(yy.min(), yy.max())
	    plt.xticks(())
	    plt.yticks(())
	    plt.title(titles[i])

	plt.show()


def decisionBDDecisionTrees(X, y):
	n_classes = 2
	plot_colors = "bry"
	plot_step = 0.02

	X = X
	y = y
	# Shuffle
	idx = np.arange(X.shape[0])
	np.random.seed(13)
	np.random.shuffle(idx)
	X = X[idx]
	y = y[idx]
	# Standardize
	mean = X.mean(axis=0)
	std = X.std(axis=0)
	X = (X - mean) / std

	clf = tree.DecisionTreeClassifier().fit(X, y)
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='s')
	plt.axis('off')
	plt.legend()
	plt.show()


def decisionBDLogRg(X, y):
	h = .02  # step size in the mesh

	logreg = linear_model.LogisticRegression(C=1).fit(X, y)

	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, m_max]x[y_min, y_max].
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1, figsize=(4, 3))
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired, marker='s')
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')

	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.show()

if __name__ == '__main__':

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:, [0, 1]], data[:, 2], test_size=0.4, random_state=0)
	

	printAccuracyLR(X_train, y_train, 0.5)
	'''
	decisionBDSVM(X_train, y_train)
	decisionBDSVM(X_test, y_test)

	decisionBDLogRg(X_train, y_train)
	decisionBDLogRg(X_test, y_test)


	decisionBDDecisionTrees(X_train, y_train)
	decisionBDDecisionTrees(X_test, y_test)
	'''



