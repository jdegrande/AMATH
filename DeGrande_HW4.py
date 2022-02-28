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
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import numpy.matlib
import sklearn.kernel_ridge
# TASK 1 #############################################################################################################
# import and preprocess the data set

# set the data path
dataPath = '/Users/jensen/Library/Mobile Documents/com~apple~CloudDocs/AMATH582/Homework/HW4/'

# import the data
votes = (np.loadtxt(dataPath + 'house-votes-84.data', delimiter = ',', dtype =object, unpack=True)).T

# construct output vector y by assigning -1, +1 to the members different parties, vector y = {435}

# construct the input vectors xj corresponding the the voting records of each member by replacing 'y' votes with +1,
# 'n' vites with -1, and '?' with 0. You DO NOT need to center and normalize the data
# matrix x {435,16}

# TASK 2: Spectral Clustering #########################################################################################
# construct unnormalized graph Laplacian matrix on X using the weight function

# take sign(q1) as your classifer and compute its classification accuracy after comparison with y


# change parameter sigma and plot the accuracy as a function of sigma


# TASK 3: Semi-Supervised Learning ###################################################################################
# consider the unnormalized Laplacian from Task 2
# given an integer M>=1 consider Laplacian embedding



# given an integer J>=1 consider teh submatrix AER^JxM and vector bER^J consisting of the first J rows of F(X) and y
# ORDERING IS IMPORTANT HERE


# Use linear regression (least squares) to find ...


# provide a table summarizing the accuracy of y_hat as your classifier for M=2,3,4,5,6 and J =5,10,20,40