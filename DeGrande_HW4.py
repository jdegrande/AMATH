# HW 4: Classifying Politicians
# Author: Jensen DeGrande
# Computational Methods for Data Analysis, AMATH 582
# Date of creation: 02/28/22
# Last edited: 03/05/22

# Problem/Purpose: Your goal is to test the performance of spectral clustering and a simple semi-supervised regression
# algorithm on the 1984 house voting records data set.

# Data: The data consists of voting records of 435 members of the House on 16 bills. There are 267 members of the
# democratic party and 168 members of the republican party. The voting record of each house member on the 16 bills will
# be our input x while the corresponding output/class y is that member's party affiliation (+/-1)
######################################################################################################################
# import packages that are needed
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from pandas import DataFrame as df
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import numpy.matlib
import sklearn.kernel_ridge
import scipy.spatial
from sklearn.linear_model import Ridge

# TASK 1 #############################################################################################################
# import and preprocess the data set
# set the data path
dataPath = '/Users/jensen/Library/Mobile Documents/com~apple~CloudDocs/AMATH582/Homework/HW4/'

# import the data
votes = (np.loadtxt(dataPath + 'house-votes-84.data', delimiter = ',', dtype =object, unpack=True)).T

# construct output vector y by assigning -1, +1 to the members different parties, vector y = {435}
party = votes[:,0]
output = np.zeros(len(party))
for i in range(0,len(party)):
    if party[i] == 'republican':
        output[i] = -1
    elif party[i] == 'democrat':
        output[i] = 1


# construct the input vectors xj corresponding the the voting records of each member by replacing 'y' votes with +1,
# 'n' vites with -1, and '?' with 0. You DO NOT need to center and normalize the data
# matrix x {435,16}
votingRecords = votes[:,1:17]
num_columns = np.shape(votingRecords)[1]
num_rows = len(votingRecords)
inputs = np.zeros([num_rows, num_columns])
for i in range(0,num_rows): # rows
    for j in range(0, num_columns): # columns
        if votingRecords[i,j] == 'y':
            inputs[i,j] = 1
        elif votingRecords[i,j] == 'n':
            inputs[i,j] = -1
        elif votingRecords[i,j] == '?':
            inputs[i,j] = 0

N =435
# TASK 2: Spectral Clustering #########################################################################################
# construct unnormalized graph Laplacian matrix on X using the weight function
# THIS IS THE GAUSSIAN
def eta(tt, rr):
#r = sigma
 val = np.exp( - (tt**2)/(2*rr**2) )
 return val.astype(float)

# pairwise distances
dist = scipy.spatial.distance_matrix(inputs, inputs, p =2)
# p is the norm 2 = euclidean

# this is what you need to tune - it will be a different number for the gaussian
r = 1
# weight will affect the shape of the clusters

# compute the weight matrix eta on pairwise distances
W = eta(dist, r)

# plot it
plt.figure(1)
plt.spy(W>=0.01)

# compute Laplacian matrices

d = np.sum(W, axis=1) # degree vector
D = np.diag(d)
L = D - W # unnormalized Laplacian with sigma = r = 1

# compute the second eigenvector
eigval, eigvec = np.linalg.eigh(L)

# we need to sort the eigenvalues and vectors
eigsorted = eigval.argsort() # sort them based on eigval index
l = eigval[eigsorted]
V = eigvec[:, eigsorted]

secondEigenvector = V[:,1]


# take sign(q1) as your classifer and compute its classification accuracy after comparison with y
# Sometimes the error will rate will be like 90% missclassified
# but if you change the sign it might become only 10% missclassified
# the classifier might choose republicans -1 instead of +1 and dems +1 instead -1
# if changing the sign of the labels fixes the problem, then the thing above happened
q1 = np.sign(secondEigenvector)
error = mean_squared_error(output,q1)

# function to calculate the number of misclassified values (probably something in python for this?)
def misclassified(true,pred):
    num_misclass = 0
    for i in range(0,len(true)):
        if pred[i] != true[i]:
            num_misclass+=1
    return num_misclass

num_misclass = misclassified(output,q1)

clusteringAccuracy = 1 - ((1/435) * num_misclass)
print(clusteringAccuracy)

# Task 2C - change parameter sigma and plot the accuracy as a function of sigma
# I'm stupid and didn't write it as a function so here is a for-loop to do it
sigma = np.linspace(0.1,4,500) # test a range of sigmas between 0 and 4
accuracy = np.zeros(len(sigma))

for i in range(0,len(sigma)):
    W_2c = eta(dist, sigma[i]) # pass each sigma value to the weight function
    d_2c = np.sum(W_2c, axis=1)  # degree vector
    D_2c = np.diag(d_2c) # degree matrix - diagonalized degree vector

    L_2c = D_2c - W_2c  # unnormalized Laplacian

    # compute the second eigenvector
    eigval_2c, eigvec_2c = np.linalg.eigh(L_2c)

    # a very manual way of sorting things? (why not use np.sort????)
    eigsorted_2c = eigval_2c.argsort()
    l_2c = eigval_2c[eigsorted_2c]
    V_2c = eigvec_2c[:, eigsorted_2c]

    secondEigenvector_2c = V_2c[:, 1]
    q1_2c = np.sign(secondEigenvector_2c)

    num_misclass_2c = misclassified(output, q1_2c)

    clusteringAccuracy_2c = (1 / 435) * num_misclass_2c
    accuracy[i] = clusteringAccuracy_2c
    #print(clusteringAccuracy)


# plot some stuff
# plt.figure(2)
# plt.scatter(sigma,accuracy, color='red')
# plt.title('Accuracy vs Sigma')
# plt.xlabel('Sigma ($\sigma$)')
# plt.ylabel('Accuracy')
# plt.text(2.5, 0.8, 'Best sigma is indicated by *')
# plt.rcParams['font.size'] = '20'
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['text.color'] = 'black'
# plt.text(3.53,0.85,"*")



plt.figure(3)
fig, ax = plt.subplots(1,2, figsize=(16,8))

ax[0].plot(np.log(l[1:None]))
ax[0].set_title('Unnormalized eigenvalues')
ax[0].set_xlabel('index')
ax[0].set_ylabel('$\lambda$')


low_acc = np.min(accuracy)
for i in range(0,len(accuracy)):
    if accuracy[i] == low_acc:
        best_sigma = sigma[i]

print(best_sigma)
#best_sigma = 1.17

# calculate the values for the best sigma so they are saved for Task 3
W_best = eta(dist, best_sigma) # pass each sigma value to the weight function
d_best = np.sum(W_best, axis=1)  # degree vector
D_best = np.diag(d_best) # degree matrix - diagonalized degree vector
L_best = D_best - W_best  # unnormalized Laplacian for the best sigma value

# compute the second eigenvector
eigval_best, eigvec_best = np.linalg.eigh(L_best)
#eigval_best and eigvel_best are not sorted

# a very manual way of sorting things? (why not use np.sort????)
eigsorted_best = eigval_best.argsort()
l_best = eigval_best[eigsorted_best]
V_best = eigvec_best[:, eigsorted_best]

secondEigenvector_best = V_best[:, 1]
q1_best = np.sign(secondEigenvector_best)
num_misclass_best = misclassified(output, q1_best)
clusteringAccuracy_best = 1 - ((1 / 435) * num_misclass_best)


# TASK 3: Semi-Supervised Learning ###################################################################################
# consider the unnormalized Laplacian from Task 2
# given an integer M>=1 consider Laplacian embedding

# build the 435xM matricx where M is the number of eigenvectors (columns) to include


# So, let me see if I roughly understand this correctly, for the semi-supervised part. The value of J is the one that
# defines the number of input and outputs you are training on, and M is how much of each eigenvector you are training
# with? And the eigenvectors are kind of like PCA where the first few are the most important?
# And we only really need the first two eigenvectors because we are mainly relying on the Fiedler vector?

M = [2, 3, 4, 5, 6]  # number of eigenvectors to include
J = [5, 10, 20, 40]  # number of members to include

# run a for-loop to go through all of the possible combinations of M and J (and fills my love for for-loops)
for i in range(0,len(M)):
    for j in range (0,len(J)):
        matrix = eigvec_best[:,0:M[i]] # matrix is size 435 x M

    # given an integer J>=1 consider the submatrix AER^JxM and vector bER^J consisting of the first J rows of F(X) and y
    # ORDERING IS IMPORTANT HERE

        A = matrix[0:J[j],:] # this is the full subset # A = 5x2
        b = output[0:J[j]] # b is just the subset of y # b = 5 matrix
        Y = output[0:M[i]]

    # Use linear regression (least squares) to find ...
    # create linear regression model
        SSLRidge = Ridge(alpha = 1e-8, fit_intercept=False) # we don't really need regularization here since K is small compared to M
        SSLRidge.fit(A, b)
        #lin = LinearRegression().fit(A, b)
        c_hat = SSLRidge.coef_

        # predict labels on entire data set
        y_pred = np.dot(matrix, c_hat)
        y_hat = np.sign(y_pred)

        num_misclass_task3 = misclassified(output,y_hat) # where output is actual labels of all the data

        SSL_accuracy =1 - ((1/435) * num_misclass_task3)
        print('M:', M[i], 'J:', J[j], 'accuracy:', SSL_accuracy)

# provide a table summarizing the accuracy of y_hat as your classifier for M=2,3,4,5,6 and J =5,10,20,40


# plot the sign q1?
xx = np.linspace(0,434,435)


q1sort = np.sort(q1_best)
newout = np.sort(output)

plt.scatter(xx,q1sort,color='red')
plt.scatter(xx,newout, color='blue',marker='*')
plt.legend(['predicted','actual'])
plt.xlabel('index')
plt.ylabel('-1 to 1 scale')
plt.yticks([-1,0,1],['republican', 'unclassified', 'democrat'] )