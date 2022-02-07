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
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[(N)*i+j,:].reshape((16, 16)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)

plot_digits(X_train, 8, "First 64 Training Features" )

print(Y_train[0:8**2])