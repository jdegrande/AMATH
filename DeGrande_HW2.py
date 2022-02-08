# HW 2: Classifying Digits
# Author: Jensen DeGrande
# Computational Methods for Data Analysis, AMATH 582
# Date of creation: 01/31/22
# Last edited: -- -- --

# Problem/Purpose: Your goal is to train a classifier to distinguish images of handwritten digits from the famous MNIST
# data set. This is a classic problem in machine learning and often times one of the first benchmarks one tries new
# algorithms on

# Data: The data is split into a training set and a test set. You will train the classifiers on the training set while
# the test set will be used for validation/evaluation of your classifiers. The training set contain 2000 instances of
# handwritten digits, the "features" are 16x16 black and white images while the labels are the corresponding digit.
# Note: the images are shaped as vectors of size 356 nd need to be reshaped for visualization. The test set only
# contains 500 instances.
######################################################################################################################
# import packages that are needed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ############### Setting up the data - I miss sections in matlab :(, wonder if there is something similar in python?
# set the data path
dataPath = '/Users/jensen/Library/Mobile Documents/com~apple~CloudDocs/AMATH582/Homework/HW2/'
# load the data
trainingSet = np.load(dataPath + 'MNIST_training_set.npy', allow_pickle=True)
testingSet = np.load(dataPath + 'MNIST_test_set.npy', allow_pickle=True)

X_train = trainingSet.item().get('features')
Y_train = trainingSet.item().get('labels')

print(X_train.shape)
print(Y_train.shape)

X_test = testingSet.item().get('features')
Y_test = testingSet.item().get('labels')

print(X_test.shape)
print(Y_test.shape)

# Plot some of the training and test sets
def plot_digits(XX, N, title):
    """Small helper function to plot N**2 digits."""
    plt.figure()
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[(N)*i+j,:].reshape((16, 16)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)

#plot_digits(X_train, 8, "First 64 Training Features" )

print(Y_train[0:8**2])

######################################################################################################################
# Task 1: Use PCA to investigate the dimensionality of Xtrain and plot the first 16 PCA modes as 16x16 images

# fig, ax = plt.subplots(4,4, figsize=(20,20))
# 256*256 makes sense
# Where Principle component 1 (top, left) accounts for the most variation and is used the most in grouping the
# handwritten digits into 0 through 9.
# pca.explained_variance_ field

# create the PCA object, train it with Xtrain, then plot the components using the same code that he has in the helper
# notebook

# from lecture 10 code
# centered_data = X_train - np.mean(X_train, axis=1)[:, None]
# dU, ds, dVt = np.linalg.svd(centered_data)
#print(dU.shape, ds.shape, dVt.shape ) # U, S, V^t


# from documentation for PCA --- If 0 < n_components < 1 and svd_solver == 'full', select the number of components
# such that the amount of variance that needs to be explained is greater than the percentage specified by n_components"
# i.e. if you give it a value n_components < 1, it finds how many components it'll take to reach the variance of
# whatever n_components is
# The number of components can either be determined by choosing the exact  number of components to keep (n_components
# is an integer >= 1) or a percent of the variance to keep (n_components is a float between 0 and 1). If you then use
# inverse_transform on the reduced dataset, you will get back an approximation of the full dataset using only the
# first n_components



# if you leave n_components_ blank - it'll use all of the data
pca = PCA()
pca.fit(X_train)

# plot using pca.components_
# UNCOMMENT LATER
# plot_digits(pca.components_, 4, 'Plotting PCA components from X train')
#
# # UNCOMMENT LATER
# plt.figure()
# plt.plot(np.log(pca.singular_values_))
# plt.plot(pca.singular_values_)
# plt.xlabel('index $j$')
# plt.ylabel('$\log(\sigma_j)$')
# plt.legend(['pca.singular_values_', 'log(pca.singular_values_)'])


######################################################################################################################
# Task 2: How many PCA modes do you need in order to approximate Xtrain up to 60%, 80%, and 90% in the Frobenius norm?
# Do you need the entire 16x16 image for each data point? No?

# need more modes to approximate greater percentage
# for the 90%
pca90 = PCA(n_components=0.9)
pca90.fit(X_train)
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)
print(pca90.components_.shape)
print(np.sum(pca90.explained_variance_ratio_)) # what is this

# for the 80%
pca80 = PCA(n_components=0.8)
pca80.fit(X_train)
print(pca80.components_.shape) # this is the num of modes
print(np.sum(pca80.explained_variance_ratio_))

# for the 60%
pca60= PCA(n_components=0.6) # this is the num of modes
pca60.fit(X_train)
print(pca60.components_.shape)
print(np.sum(pca60.explained_variance_ratio_))


# maybe reconstruct them to prove this?








# Task 3: Train a classifier to distinguish the digits 1 and 8



# To highlight what I mean, here are the general steps for the task of building a binary classifier between 1s and 8s are
#
# First perform PCA on all of the training images (all digits, not just 1s and 8s) and take the first 16 modes.
# Other choices could be made when building a classfier, but for this homework task you are told to use the first 16
# components from performing PCA on all digits so that we can easily check that you got the expected results.
# Next pull the relevant training images (1s and 8s) in order to build the training set for your classfier. Also pull
# the relevant training labels and convert to -1 and +1 instead of 1s and 8s. Project the training set onto the 16 PCA
# components. Now your training images are each represented by a point in the 16-dimensional component space. Train your
# classifier to associate points in 16 dimensional space with either the label +1 or =1 by fitting it to the training
# data and training labels.