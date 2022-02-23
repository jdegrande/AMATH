# HW 3: Qualifying Red Wine
# Author: Jensen DeGrande
# Computational Methods for Data Analysis, AMATH 582
# Date of creation: 02/22/22
# Last edited: -- -- --

# Problem/Purpose: Your goal is to develop an algorithm that predicts the quality of wine from a series of chemical
# measurements. This will be used to price the wines

# Data: The data consists of a training set of 1115 instances (different types of wine that have been measured) and a
# test data set of 479 instances. Each instance has 11 attributes (features) that are outlined in a wine_description.txt
# The corresponding output to each set of features is the quality of the wine on a scale of 0 to 10. You are also given
# the lab measurements for a batch of five new wines where we must predict the qualities for.
######################################################################################################################
# import packages that are needed
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import numpy.matlib
import sklearn.kernel_ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

# ############### Setting up the data
# set the data path
dataPath = '/Users/jensen/Library/Mobile Documents/com~apple~CloudDocs/AMATH582/Homework/HW3/'

# import the data
training = np.loadtxt(dataPath + 'wine_training.csv', delimiter=',')
testing = np.loadtxt(dataPath + 'wine_test.csv', delimiter=',')

# convert to np array
training = np.array(training)
testing = np.array(testing)



# separate the data into features X and output Y
X_train = training[:, 0:11]
Y_train = training[:,11]

X_test = testing[:, 0:11]
Y_test = testing[:,11]

# generate some plots to visualize the data set

# fig, ax = plt.subplots(2,3, figsize=(24,12))
#
# for j in range(2):
#   for i in range(3):
#
#     ax[j][i].scatter( X_train[:, i+ j*3], Y_train )
#     ax[j][i].set_xlabel('x_'+str(i + j*3), fontsize=20)


# normalize and center the input data (xj's) and outputs (yj's)
# they should have a mean 0 and standard deviation of 1
# (look at lecture 16 code)

# Next we normalize and center the training set

X_train_N = X_train.shape[0]

X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

# must normalize it
X_train_normal = (X_train - np.matlib.repmat(X_train_mean, X_train_N, 1))/np.matlib.repmat(X_train_std, X_train_N, 1)

Y_train_N = Y_train.shape[0]

Y_train_mean = np.mean(Y_train, axis=0)
Y_train_std = np.std(Y_train, axis=0)

Y_train_normal = (Y_train - Y_train_mean)/Y_train_std

print(X_train_normal.shape)
print(Y_train_normal.shape)

# fig, ax = plt.subplots(2,3, figsize=(24,12))
#
# for j in range(2):
#   for i in range(3):
#
#     ax[j][i].scatter( X_train_normal[:, i+ j*3], Y_train_normal )
#     ax[j][i].set_xlabel('x_'+str(i + j*3), fontsize=20)

# while we're here we also normalize and center the test set.
# NOTE: the shift and scaling here are those computed on the training set
X_test_N = X_test.shape[0]

# do the same with the test set but use the TRAINING MEAN and STD
X_test_normal = (X_test - np.matlib.repmat(X_train_mean, X_test_N, 1))/np.matlib.repmat(X_train_std, X_test_N, 1)
Y_test_normal = (Y_test - Y_train_mean)/Y_train_std


# Task 1: Use linear regression (least squares) to fit a linear model to the training set
# linear regression

lin = LinearRegression().fit(X_train_normal, Y_train_normal)
lin_train_pred = lin.predict(X_train_normal)
lin_test_pred = lin.predict(X_test_normal)
lin_train_mse = mean_squared_error(Y_train_normal, lin_train_pred)
lin_test_mse = mean_squared_error(Y_test_normal, lin_test_pred)

#print(lin.score(X_train_normal,Y_train_normal))

print(lin_train_mse)
print(lin_test_mse)

# kernel ridge regression for Gaussian
sigma = 0.5

# create the testing parameters for the gaussian (rbf) estimator
gauss_alpha = 2**np.linspace(-5,5,10)
gauss_gamma = 1/(2*(gauss_alpha**2))
#scoring = ['gauss_alpha', 'gauss_gamma']

parameters = {'kernel':'rbf', 'alpha':gauss_alpha, 'gamma':gauss_gamma}
svc = sklearn.kernel_ridge.KernelRidge() # create estimator
KRR = GridSearchCV(svc, parameters)
#KRR = sklearn.kernel_ridge.KernelRidge(kernel='rbf', alpha = gauss_alpha,gamma=gauss_gamma) # gamma = 1/(2**(sigma**2))
KRR.fit(X_train_normal, Y_train_normal)



#clf = svm.SVC(kernel='linear', C=1, random_state=0)
#scores = cross_validate(KRR, X_train_normal, Y_train_normal, scoring=scoring,cv=10,return_train_score=True)
#sorted(scores.keys())



# use our fitted model to predict the Y values on the test set
Y_pred_normal = KRR.predict(X_test_normal)

# MSE
#KRR_train_mse = mean_squared_error(Y_train_normal, Y_pred_normal)
KRR_test_mse = mean_squared_error(Y_test_normal, Y_pred_normal)
print(KRR_test_mse)

# plot the predicted values of Y against the test set
# fig, ax = plt.subplots(6,2, figsize=(12,6))
#
# for j in range(6):
#   for i in range(2):
#
#     ax[j][i].scatter( X_test_normal[:, i+ j*2], Y_test_normal, color='r', label='Test' )
#     ax[j][i].scatter( X_test_normal[:, i+ j*2], Y_pred_normal, color='b', label='Prediction' )
#     ax[j][i].set_xlabel('x_'+str(i + j*2), fontsize=10)
#
# plt.legend(fontsize=10)



# kernel ridge regression for Laplacian
# kernel ridge regression for Gaussian
sigma_lap = 0.5

KRR_lap = sklearn.kernel_ridge.KernelRidge(kernel='laplacian', alpha = 0.9,gamma=1/(2*sigma**2)) # gamma = 1/(2**sigma)
KRR_lap.fit(X_train_normal, Y_train_normal)
#score = KRR_lap.score(X_test_normal,Y_test_normal)

# use our fitted model to predict the Y values on the test set
Y_pred_normal_lap = KRR_lap.predict(X_test_normal)

KRR_lap_test_mse = mean_squared_error(Y_test_normal, Y_pred_normal_lap)
print(KRR_lap_test_mse)

# kernel regression is sensitive to the choice of regularization parameter (lambda) and length scale of the kernel
# (sigma). Prototype your code on a subset of the data and find a ball-park value for the parameters before running the
# full simulation

# GridSearchCV
