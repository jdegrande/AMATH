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
# Note: the images are shaped as vectors of size 256 nd need to be reshaped for visualization. The test set only
# contains 500 instances.
######################################################################################################################
# import packages that are needed
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
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
            ax[i, j].imshow(XX[(N) * i + j, :].reshape((16, 16)), cmap="Greys")
            ax[i, j].axis("off")
    fig.suptitle(title, fontsize=24)


# plot_digits(X_train, 8, "First 64 Training Features" )

print(Y_train[0:8 ** 2])

######################################################################################################################
# Task 1: Use PCA to investigate the dimensionality of Xtrain and plot the first 16 PCA modes as 16x16 images
# create the PCA object, train it with Xtrain, then plot the components using plot_digits

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
pca.fit(X_train)  # .fit automatically centers the data
# X_train_standard = StandardScaler().fit_transform(X_train) -- don't need to scale b/c they are all pixels

plt.figure()
plt.rcParams['font.size'] = '20'
plt.plot(pca.singular_values_)
#plt.plot(np.log(pca.explained_variance_))
#plt.vlines(x=150, ymin=-5, ymax=100, colors='r')
plt.title('Dimensionality Analysis')
plt.xlabel('index $j$')
plt.ylabel('$\sigma_j$')
plt.legend(['pca.singular_values_'])

# plot the first 16
plot_digits(pca.components_, 4, 'First 16 PCA Modes')

# reconstruct it
pca16 = PCA(16)
pca16.fit(X_train)
transformed_X = pca.transform(X_train)
# inverse_X = pca.inverse_transform(transformed_X)
# then use plot_digits to plot these images
# plot_digits(pca16.components_, 4, 'First 16 PCA Modes')
# plot_digits(transformed_X,4,'Transformed')

######################################################################################################################
# Task 2: How many PCA modes do you need in order to approximate Xtrain up to 60%, 80%, and 90% in the Frobenius norm?
# Do you need the entire 16x16 image for each data point?
# from documentation:  singular_values_ndarray of shape (n_components,) = The singular values corresponding to each of
# the selected components. The singular values are equal to the 2-norms of the n_components variables in the
# lower-dimensional space.

modesSum = np.sum(pca.singular_values_ ** 2)
normTotal = np.sqrt(modesSum)
print(normTotal)

norm90 = normTotal * .9
norm80 = normTotal * .8
norm60 = normTotal * .6

def pca_modes(percent):
    pca_loop = PCA(n_components=1)  # start at the first mode to initialize loop
    pca_loop.fit(X_train)
    i = 1
    while (np.sqrt(sum(pca_loop.singular_values_ ** 2))) < (normTotal * percent): # singular values are the sigmas
        i += 1
        pca_loop = PCA(n_components=i)
        pca_loop.fit(X_train)
    return i


print(pca_modes(0.9)) #14
print(pca_modes(0.8)) #7
print(pca_modes(0.6)) #3

modes= []
num = np.linspace(0,1.0,num =100)

for i in num:
    value = pca_modes(i)
    modes.append(value)


# make the plots with the cutoff value set at each one
plt.figure()
plt.plot(num,modes)
plt.plot(0.6,3, 'k*')
plt.text(0.56,8,'(0.6,3)')
plt.plot(0.8,7, 'k*')
plt.text(0.76,13,'(0.8,7)')
plt.plot(0.9,14, 'k*')
plt.text(0.86,1,'(0.9,14)')
plt.title('Number of PCA Modes for Forbenius Norm Approximation')
plt.xlabel('Percentage of Forbenius Norm')
plt.ylabel('Number of PCA modes')


# Task 3: Train a classifier to distinguish the digits 1 and 8
# From canvas discussion:
# To highlight what I mean, here are the general steps for the task of building a binary classifier between 1s and 8s are
# First perform PCA on all of the training images (all digits, not just 1s and 8s) and take the first 16 modes.
# Other choices could be made when building a classfier, but for this homework task you are told to use the first 16
# components from performing PCA on all digits so that we can easily check that you got the expected results.
# Next pull the relevant training images (1s and 8s) in order to build the training set for your classfier. Also pull
# the relevant training labels and convert to -1 and +1 instead of 1s and 8s. Project the training set onto the 16 PCA
# components. Now your training images are each represented by a point in the 16-dimensional component space. Train your
# classifier to associate points in 16 dimensional space with either the label +1 or =1 by fitting it to the training
# data and training labels.

# get indices with np.where
# then we can do x18 = X_train[indices] == this will take all the rows that correspond to it

# functions for Task 3 ###############################################################################################
# Write a function that extracts the features and labels of the digits 1 and 8 from the training data set X(1,8)
# and Y(1,8) --- pass X_train and Y_train
def extract(X, Y, num1, num2):
    count = 0
    # extractX = np.array([])
    extractX = np.empty((1, 256))
    extractY = np.array([])
    for i in range(0, len(Y)):
        if (Y[i] == num1) or (Y[i] == num2): # if value is equal to num1 or num2
            # pull index value associated with index i
            extractY = np.append(extractY, Y[i])
            extractX = np.vstack((extractX, X[i, :]))  # pull the row corresponding to that index
            count += 1
            #print(count)
    return extractX, extractY

# function for relabeling values with -1 and 1
def new_label(labels, num1, num2):
    for i in range(0, len(labels)):
        if labels[i] == num1:
            labels[i] = -1
        elif labels[i] == num2:
            labels[i] = 1
    return labels


[X1_8_preflip, Y1_8] = extract(X_train, Y_train, 1, 8)  # this is the training set for our classifier
X1_8 = np.delete(X1_8_preflip, 0, 0)

# project X1_8 onto the first 16 PCA modes of X_train computed previously (Task 1)
atrain = pca16.transform(X1_8)

# reassign labels - create btrain
btrain = new_label(Y1_8, 1, 8)

# Use Ridge regression or least squares to train a predictor for the vector Btrain by linearly combining
# the columns of Atrain
# 1. Create a ridge regressor
# 2. Fit it on the training data
# 3. Predict the training and testing data using the fitted model
# 4. Use the predictions to calculate the MSE


afunmodel = linear_model.RidgeCV()  # alpha=1) # alpha is our lambda
# ridgeCV optimizes for us so it will provide the best model for each classifier
afunmodel.fit(atrain, btrain)
betas = afunmodel.predict(atrain)  # or is this (atest)

MSEtrain18 = sklearn.metrics.mean_squared_error(btrain, betas)
# MSE atrain x weight that comes out of classifier

newMSE = (1/len(btrain)) * np.linalg.norm(betas - btrain)**2



# Let's repeat the process for testing: "here's some other data, let's see how well we trained our PCA by cross
# checking with y_test"
# pca.transform(X_test) and then RidgeClassifier.predict(X_test).
# Test will have more error because the classifier you "fit" or "trained" was on the training data, so in some sense
# it over fits the training data
# Test data will give you an idea of how a new sample will perform

[Xtest1_8_pre, Ytest1_8] = extract(X_test, Y_test, 1, 8)  # this is the training set for our classifier
Xtest1_8 = np.delete(Xtest1_8_pre, 0, 0)
atest = pca16.transform(Xtest1_8)
btest = new_label(Ytest1_8, 1, 8)

betasTest = afunmodel.predict(atest)
MSEtest18 = sklearn.metrics.mean_squared_error(btest, betasTest) # (ytrue, ypred)

rounded = np.round(betasTest)

# Let's plot some figures
plt.figure()
for i in range(0,len(rounded)):
    if rounded[i] == -1:
        plt.plot(i,betasTest[i],'kx')
    elif rounded[i] == 1:
        plt.plot(i,betasTest[i], 'ko')
plt.hlines(y=-1, xmin=0, xmax=len(betasTest),colors='r')
plt.hlines(y=1, xmin=0, xmax=len(betasTest),colors='r')
plt.title('Test Data (1,8) Performance')
plt.xlabel('index $j$')
plt.ylabel('Digit Label')
plt.legend(['predicted 1s', 'predicted 8s'])



# Task 4: Use your code from step 3 to train classifiers for the pairs of the digits (3,8) and (2,7) and report the
# training and test MSE's. Can you explain the performance variations?

# let's repeat the process for (3,8)
[X3_8temp, Y3_8] = extract(X_train, Y_train, 3, 8)  # this is the training set for our classifier
X3_8 = np.delete(X3_8temp, 0, 0)

atrain38 = pca16.transform(X3_8) # append a row of 1's to the beginning ??

# reassign labels - create btrain
btrain38 = new_label(Y3_8, 3, 8)

# create the model to classify 3 and 8
model38 = linear_model.RidgeCV()  # alpha=1) # alpha is our lambda
# use pca.transform and ridge.predict on the test data
model38.fit(atrain38, btrain38)
betas38 = model38.predict(atrain38)  # or is this (atest)

MSEtrain38 = sklearn.metrics.mean_squared_error(btrain38, betas38)

# let's do testing for values 3 and 8
[Xtest3_8temp, Ytest3_8] = extract(X_test, Y_test, 3, 8)  # this is the training set for our classifier
Xtest3_8 = np.delete(Xtest3_8temp, 0, 0)

atest38 = pca16.transform(Xtest3_8)
btest38 = new_label(Ytest3_8, 3, 8)

betasTest38 = model38.predict(atest38)
MSEtest38 = sklearn.metrics.mean_squared_error(btest38, betasTest38)

# Let's make a performance plot
rounded38 = np.round(betasTest38)

# Let's plot some figures
plt.figure()
for i in range(0,len(rounded38)):
    if (betasTest38[i] < 0):
        plt.plot(i,betasTest38[i],'kx')
    elif (betasTest38[i] >= 0):
        plt.plot(i,betasTest38[i], 'ko')
plt.hlines(y=-1, xmin=0, xmax=len(betasTest38),colors='r')
plt.hlines(y=1, xmin=0, xmax=len(betasTest38),colors='r')
plt.title('Test Data (3,8) Performance')
plt.xlabel('index $j$')
plt.ylabel('Digit Label')
plt.legend(['predicted 3s', 'predicted 8s'])



# let's repeat the process for (2,7)
[X2_7temp, Y2_7] = extract(X_train, Y_train, 2, 7)  # this is the training set for our classifier
X2_7 = np.delete(X2_7temp, 0, 0)

atrain27 = pca16.transform(X2_7) # append a row of 1's to the beginning ??

# reassign labels - create btrain
btrain27 = new_label(Y2_7, 2, 7)

# create the model to classify 3 and 8
model27 = linear_model.RidgeCV()  # alpha=1) # alpha is our lambda
# use pca.transform and ridge.predict on the test data
model27.fit(atrain27, btrain27)
betas27 = model27.predict(atrain27)  # or is this (atest)

MSEtrain27 = sklearn.metrics.mean_squared_error(btrain27, betas27)

# let's do testing for values 3 and 8
[Xtest2_7temp, Ytest2_7] = extract(X_test, Y_test, 2, 7)  # this is the training set for our classifier
Xtest2_7 = np.delete(Xtest2_7temp, 0, 0)

atest27 = pca16.transform(Xtest2_7)
btest27 = new_label(Ytest2_7, 2, 7)

betasTest27 = model27.predict(atest27)
MSEtest27 = sklearn.metrics.mean_squared_error(btest27, betasTest27)


rounded27 = np.round(betasTest27)

# Let's plot some figures
plt.figure()
for i in range(0,len(rounded27)):
    if rounded27[i] == -1:
        plt.plot(i,betasTest27[i],'kx')
    elif rounded27[i] == 1:
        plt.plot(i,betasTest27[i], 'ko')
plt.hlines(y=-1, xmin=0, xmax=len(betasTest27),colors='r')
plt.hlines(y=1, xmin=0, xmax=len(betasTest27),colors='r')
plt.title('Test Data (2,7) Performance')
plt.xlabel('index $j$')
plt.ylabel('Digit Label')
plt.legend(['predicted 2s', 'predicted 7s'])