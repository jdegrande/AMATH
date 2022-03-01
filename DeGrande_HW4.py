# HW 4: Classifying Politicians
# Author: Jensen DeGrande
# Computational Methods for Data Analysis, AMATH 582
# Date of creation: 02/28/22
# Last edited: -- -- --

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
# weight will effect the shape of the clusters

# compute the weight matrix eta on pairwise distances
W = eta(dist, r)

# plot it
plt.spy(W>=0.01)

# visualize the graph using NetworkX
#
# import networkx as nx
#
# G = nx.Graph()
#
# for i in range(N):
#   for j in range(N):
#     if i != j and W[i,j] != 0 :
#       G.add_edge(i, j, weight=W[i,j])

# fig, ax = plt.subplots(1,2, figsize=(16,8))
#
# ax[0].scatter(inputs[:,0], inputs[:,1], color='b')
# ax[0].set_aspect('equal')
# ax[0].set_xlabel('$x_1$')
# ax[0].set_ylabel('$x_2$')
# ax[0].set_title('Input (Unlabelled) data')
#
#
# nx.draw_networkx_nodes(G, inputs, node_size=100, ax = ax[1])
# nx.draw_networkx_edges(G, inputs, ax = ax[1])
# ax[1].set_aspect('equal')
# ax[1].set_title('Proximity Graph')


# compute Laplacian matrices

d = np.sum(W, axis=1) # degree vector

D = np.diag(d)
Ds = np.diag(1/np.sqrt(d))
Di = np.diag(1/d)


L = D - W # unnormalized

# compute the second eigenvector
eigval, eigvec = np.linalg.eigh(L)

# we need to sort the eigenvalues and vectors

# a very manual way of sorting things (why not use np.sort????)
eigsorted = eigval.argsort()
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

def misclassified(true,pred):
    num_misclass = 0
    for i in range(0,len(true)):
        if true[i] != pred[i]:
            num_misclass+=1
    return num_misclass

num_misclass = misclassified(output,q1)

clusteringAccuracy = (1/435) * num_misclass
print(clusteringAccuracy)

# Task 2C - change parameter sigma and plot the accuracy as a function of sigma
# I'm stupid and didn't write it as a function so here is a for-loop to do it
sigma = np.linspace(0.01,4,100)
accuracy = np.zeros(len(sigma))

for i in range(0,len(sigma)):
    W_2c = eta(dist, sigma[i]) # pass each sigma value to the weight function

    d_2c = np.sum(W_2c, axis=1)  # degree vector

    D_2c = np.diag(d_2c)
    Ds_2c = np.diag(1 / np.sqrt(d_2c))
    Di_2c = np.diag(1 / d_2c)

    L_2c = D_2c - W_2c  # unnormalized

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



# TASK 3: Semi-Supervised Learning ###################################################################################
# consider the unnormalized Laplacian from Task 2
# given an integer M>=1 consider Laplacian embedding



# given an integer J>=1 consider teh submatrix AER^JxM and vector bER^J consisting of the first J rows of F(X) and y
# ORDERING IS IMPORTANT HERE


# Use linear regression (least squares) to find ...


# provide a table summarizing the accuracy of y_hat as your classifier for M=2,3,4,5,6 and J =5,10,20,40