# HW 3: Qualifying Red Wine
# Author: Jensen DeGrande
# Computational Methods for Data Analysis, AMATH 582
# Date of creation: 02/22/22
# Last edited: 02/25/22

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
newbatchX = np.loadtxt(dataPath+ 'wine_new_batch.csv', delimiter=',')

# convert to np array
training = np.array(training)
testing = np.array(testing)
newbatchX = np.array(newbatchX)


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
# normalize X_train
X_train_N = X_train.shape[0]

X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_normal = (X_train - np.matlib.repmat(X_train_mean, X_train_N, 1))/np.matlib.repmat(X_train_std, X_train_N, 1)

# normalize the Y_train
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
newbatchX_N = newbatchX.shape[0]

# do the same with the test set but use the TRAINING MEAN and STD
X_test_normal = (X_test - np.matlib.repmat(X_train_mean, X_test_N, 1))/np.matlib.repmat(X_train_std, X_test_N, 1)
Y_test_normal = (Y_test - Y_train_mean)/Y_train_std
newbatchX_normal = (newbatchX - np.matlib.repmat(X_train_mean, newbatchX_N,1))/np.matlib.repmat(X_train_std,newbatchX_N,1)


# Task 1: Use linear regression (least squares) to fit a linear model to the training set
# linear regression

# create linear regression model
lin = LinearRegression().fit(X_train_normal, Y_train_normal)
# make predictions for training and testing sets (normalized)
lin_train_pred = lin.predict(X_train_normal)
lin_test_pred = lin.predict(X_test_normal)
# compute MSE
lin_train_mse = mean_squared_error(Y_train_normal, lin_train_pred)
lin_test_mse = mean_squared_error(Y_test_normal, lin_test_pred)
# predict the values for the new batch
lin_newbatch_pred = lin.predict(newbatchX_normal)
lin_newbatch_pred = (lin_newbatch_pred*Y_train_std) + Y_train_mean# denormalize to get actual values
lin_newbatch_pred_rounded = np.round(lin_newbatch_pred)


#print(lin.score(X_train_normal,Y_train_normal))

print('linear train')
print(lin_train_mse)
print('linear test')
print(lin_test_mse)

# kernel ridge regression for Gaussian
#sigma = 0.5

# create the testing parameters for the gaussian (rbf) estimator
# make it a linspace of values to test so we can cross validate and determine the best parameters
sgm = np.linspace(-3,3,10)
lmbd = np.linspace(-3,3,10)

gauss_alpha = 2**lmbd
#gauss_gamma = 1/(2*(gauss_alpha**2))
gauss_gamma = 1/(2*(2**sgm)**2)
# gauss_alpha = 2**np.linspace(-5,5,10)
# #gauss_gamma = 1/(2*(gauss_alpha**2))
# gauss_gamma = 1/(2*(2**gauss_alpha)**2)

parameters = {'kernel':['rbf'], 'alpha':(gauss_alpha), 'gamma':(gauss_gamma)}
svc = sklearn.kernel_ridge.KernelRidge() # create estimator
# use GridSearchCV and the defined estimator and parameters to determine the best alpha and gamma - obtain train mse
KRR = GridSearchCV(svc, parameters, cv=10, return_train_score=True)
#KRR = sklearn.kernel_ridge.KernelRidge(kernel='rbf', alpha = gauss_alpha,gamma=gauss_gamma) # gamma = 1/(2**(sigma**2))
KRR.fit(X_train_normal, Y_train_normal)  # train the optimized model
#sorted(KRR.cv_results_.keys())

optimized = KRR.best_estimator_
print(optimized)

alpha = KRR.best_params_.get("alpha")
gamma = KRR.best_params_.get("gamma")

rbf_lmbd = np.log2(alpha)
rbf_sig = np.log2(gamma)

# use our fitted model to predict the Y for train set
Ytrain_pred = KRR.predict(X_train_normal)
# use our fitted, best parameter model to predict the Y values on the test set
Ytest_pred = KRR.predict(X_test_normal)
# use our fitted, best parameter model to predict the Y values on the new batch
Ynew_pred = KRR.predict(newbatchX_normal)
Ynew_pred = (Ynew_pred*Y_train_std)+ Y_train_mean # denormalize to get actual values
Ynew_pred_rounded = np.round(Ynew_pred)


# MSE
KRR_train_mse = mean_squared_error(Y_train_normal, Ytrain_pred)
KRR_test_mse = mean_squared_error(Y_test_normal, Ytest_pred)
print('rbf train')
print(KRR_train_mse)
print('rbf test')
print(KRR_test_mse)

# # plot the predicted values of Y against the test set
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
#sigma_lap = 0.5

# first let's run the search on a range of -5 to 5 with resolution of 10 to intially assess where a good value might be
# then let's rerun the search
# Rerun the grid search, but instead of a grid centered on zero with a range of 4 on each side, center it on the value
# you found in your first search, and lower the range so say you got a value of 2.5, your grid search would be like
# 1.5 to 3.5 or even finer, up to you, just don't make the range smaller than one box from the previous search, or you
# could miss the maximum

# define the parameters as a linspace to find the best values for each
lap_sgm = np.linspace(-1,3,10) #0,4 # 1,3
lap_lmbd = np.linspace(-1,3,10) #-3,1 # -3,1

lap_alpha = 2**lap_lmbd
lap_gamma = 1/(2**lap_sgm)


parameters_lap = {'kernel':['laplacian'], 'alpha':(lap_alpha), 'gamma':(lap_gamma)}
lap = sklearn.kernel_ridge.KernelRidge() # create estimator
# use GridSearchCV and the defined estimator and parameters to determine the best alpha and gamma - obtain train mse
KRR_lap = GridSearchCV(lap, parameters_lap, cv=10, return_train_score=True,scoring='neg_mean_squared_error')
KRR_lap.fit(X_train_normal, Y_train_normal)


#KRR_lap = sklearn.kernel_ridge.KernelRidge(kernel='laplacian', alpha = 0.9,gamma=1/(2*sigma**2)) # gamma = 1/(2**sigma)
#KRR_lap.fit(X_train_normal, Y_train_normal)
#score = KRR_lap.score(X_test_normal,Y_test_normal)

optLap = KRR_lap.best_estimator_
print(optLap)

alpha_lap = KRR_lap.best_params_.get("alpha")
gamma_lap = KRR_lap.best_params_.get("gamma")


lap_lmbd = np.log2(alpha_lap)
lap_sig = np.log2(gamma_lap)

# use our fitted model to predict the Y values on the train set
Ytrain_pred_lap = KRR_lap.predict(X_train_normal)
# use our fitted model to predict the Y values on the test set
Ytest_pred_lap = KRR_lap.predict(X_test_normal)
# use our fitted, best parameter model to predict the Y values on the new batch
Ynew_pred_lap_norm = KRR_lap.predict(newbatchX_normal)
Ynew_pred_lap = (Ynew_pred_lap_norm*Y_train_std)+ Y_train_mean # denormalize to get actual values
Ynew_pred_lap_rounded = np.round(Ynew_pred_lap)



KRR_lap_train_mse = mean_squared_error(Y_train_normal, Ytrain_pred_lap)
KRR_lap_test_mse = mean_squared_error(Y_test_normal, Ytest_pred_lap)

print('lap train mse')
print(KRR_lap_train_mse)
print('lap test mse')
print(KRR_lap_test_mse)


print('linear predictions')
print(lin_newbatch_pred)
print(lin_newbatch_pred_rounded)

print('rbf predictions')
print(Ynew_pred)
print(Ynew_pred_rounded)

print('Laplacian predictions')
print(Ynew_pred_lap)
print(Ynew_pred_lap_rounded)

print('done!')


