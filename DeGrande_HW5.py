# HW 5: Compressed Image Recovery
# Author: Jensen DeGrande
# Computational Methods for Data Analysis, AMATH 582
# Date of creation: 03/09/22
# Last edited: -- -- --

# Problem/Purpose: The purpose of this assignment is to recover an image from limited observations of its pixels. We
# will consider Rene Magritte's "The Son of Man" and a corrupted variant in order to understand image recovery. Our goal
# is to recover the original image from the corrupted version.

# Data: The data files include SonOfMan.png, UnknownImage.py, and a helper code to start with plotting and intrepretting
# the data, including rescaling the original iamge to a lower resolution to make the computations manageable and more
# cost effective in terms of computational time and effort.
######################################################################################################################
# import packages that are needed
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import skimage as ski

import skimage.io
import skimage.transform

import scipy.fftpack as spfft  # for discrete cosine transform

# TASK 1  - Image Compression #########################################################################################
# The goal is to investigate the compressibility of the discrete cosine transform (DCT) of SonOfMan.png. This transform
# is analogous to taking the real part of the FFT of f i.e. writing the signal as a sum of cosines. The inverse DCT
# transform reconstructs the signal f fromt the DCT(f). The 2D DCT is defined analogously to 2D FFT by successviely
# applying the 1D DCT to the rows and columns of a 2D image.
# perform the following on the rescaled 53x41 image, not the original
#

# import and preprocess the data set
# set the data path and import the data
dataPath = '/Users/jensen/Library/Mobile Documents/com~apple~CloudDocs/AMATH582/Homework/HW5/'
img_path = dataPath + 'SonOfMan.png'

# read image
img_og = ski.io.imread(img_path)

# convert to grayscale and visualize
img_og = ski.color.rgb2gray(img_og)

# fig, ax = plt.subplots(1, 2, figsize=(20,10))
# ax[0].imshow(img_og, cmap = 'gray')
# ax[0].set_title("Original image")

print("Original size: ", img_og.shape)

# rescale the image using the code from the helper code
# resize image
img = ski.transform.rescale(img_og, 0.18, anti_aliasing=False)

print("Rescaled size: ", img.shape)

# ax[1].imshow(img, cmap='gray')
# ax[1].set_title("Rescaled image")


numRows = img.shape[0]
numCols = img.shape[1]

pixels = numRows * numCols


# 1a) construct the forward and inverse DCT matrices for the image
# construct forward and inverse DCT matrices

# construct DCT matrix
def construct_DCT_Mat(Nx, Ny):
    # input : Nx number of columns of image
    #         Ny number of rows of image
    # output: D DCT matrix mapping image.flatten() to DCT(image).flatten()

    Dx = spfft.dct(np.eye(Nx), axis=0, norm='ortho')
    Dy = spfft.dct(np.eye(Ny), axis=0, norm='ortho')
    D = np.kron(Dy, Dx)

    return D


# construct inverse DCT matrix
def construct_iDCT_Mat(Nx, Ny):
    # input : Nx number of columns of image
    #         Ny number of rows of image
    # output: iD iDCT matrix mapping DCT(image).flatten() to image.flatten()

    Dx = spfft.idct(np.eye(Nx), axis=0, norm='ortho')
    Dy = spfft.idct(np.eye(Ny), axis=0, norm='ortho')
    D = np.kron(Dy, Dx)

    return D


newDCT = construct_DCT_Mat(numCols, numRows)
newiDCT = construct_iDCT_Mat(numCols, numRows)

f = np.ndarray.flatten(img)
index = np.arange(0, pixels)

DCT_F = np.matmul(newDCT, f)

# UNCOMMENT THIS FOR REPORT
# 1b) plot DCT(f) and investigate the compressibility. Do you see a lot of large coefficients??
# plt.figure(3)
# plt.scatter(index,DCT_F)
# plt.title('DCT(F)')
# plt.xlabel('index')
# plt.ylabel('DCT(F)')


# 1c) reconstruct and plot the image after thresholding its DCT to keep the top 5,10,20, and 40 percent of DCT
# coefficients

DCT_F_copy = np.copy(DCT_F)
# DCT_F_abs = np.abs(DCT_F_copy)
DCT_F_sort = DCT_F_copy[np.argsort(-DCT_F_copy)]

percentages = [0.05, 0.1, 0.2, 0.4]  # take top 5%, 10%, 20%, and 40%

dctF5 = np.copy(DCT_F)
dctF10 = np.copy(DCT_F)
dctF20 = np.copy(DCT_F)
dctF40 = np.copy(DCT_F)

# plt.figure()
# fig, ax = plt.subplots(2, 2, figsize=(8, 8))

# too frustrated to make the for-loop work
# for i in range(0, len(percentages)):
#   file = 'dctF' + str(int(percentages[i] *100))
#   percent = round(percentages[i] * len(DCT_F_sort))
#   print('For ' + str(percentages[i]) + ' there are ' + str(percent) + ' values to include')
#   threshold = DCT_F_sort[percent]
#   # testing = np.percentile(DCT_F_sort)
#   # fivePercent = np.zeros(pixels)
#   # fivePercent[0:percent] = DCT_F_sort[0:percent]
#   for j in range(0, len(DCT_F)):
#       if abs(file[j]) < threshold:
#           file[j] = 0
#       else:
#           file[j] = file[j]
#   reconstruction = np.dot(newiDCT, file)
#   # reconstruction = np.dot(newiDCT,fivePercent)
#   # reconstruction = np.dot(DCT_F_sort,newiDCT)
#   shaped = np.reshape(reconstruction, (53, 41))
#
#   plt.figure(i)
#   plt.imshow(shaped, cmap='gray')
#   plt.title(str(percentages[i] * 100) + "% image")

    # ax[p, j].imshow(shaped, cmap='gray')
    # ax[p, j].axis("off")
    # ax[p, j].set_title(str(percent) + " image")
# fig.suptitle(title, fontsize=24)

# for 5%
percent = round(0.05 * len(DCT_F_sort))
print('For ' + str(0.05) + ' there are ' + str(percent) + ' values to include')
threshold = DCT_F_sort[percent]
# testing = np.percentile(DCT_F_sort)
# fivePercent = np.zeros(pixels)
# fivePercent[0:percent] = DCT_F_sort[0:percent]
for j in range(0, len(DCT_F)):
    if abs(dctF5[j]) < threshold:
        dctF5[j] = 0
    else:
        dctF5[j] = dctF5[j]
reconstruction = np.dot(newiDCT, dctF5)
shaped = np.reshape(reconstruction, (53, 41))

plt.figure(2)
plt.imshow(shaped, cmap='gray')
plt.title(str(0.05 * 100) + "% image")

# for 10%
percent = round(0.1 * len(DCT_F_sort))
print('For ' + str(0.1) + ' there are ' + str(percent) + ' values to include')
threshold = DCT_F_sort[percent]
# testing = np.percentile(DCT_F_sort)
# fivePercent = np.zeros(pixels)
# fivePercent[0:percent] = DCT_F_sort[0:percent]
for j in range(0, len(DCT_F)):
    if abs(dctF10[j]) < threshold:
        dctF10[j] = 0
    else:
        dctF10[j] = dctF10[j]
reconstruction = np.dot(newiDCT, dctF10)
# reconstruction = np.dot(newiDCT,fivePercent)
# reconstruction = np.dot(DCT_F_sort,newiDCT)
shaped = np.reshape(reconstruction, (53, 41))

plt.figure(3)
plt.imshow(shaped, cmap='gray')
plt.title(str(0.1 * 100) + "% image")




# for 20%
percent = round(0.2 * len(DCT_F_sort))
print('For ' + str(0.2) + ' there are ' + str(percent) + ' values to include')
threshold = DCT_F_sort[percent]
# testing = np.percentile(DCT_F_sort)
# fivePercent = np.zeros(pixels)
# fivePercent[0:percent] = DCT_F_sort[0:percent]
for j in range(0, len(DCT_F)):
    if abs(dctF20[j]) < threshold:
        dctF20[j] = 0
    else:
        dctF20[j] = dctF20[j]
reconstruction = np.dot(newiDCT, dctF20)
# reconstruction = np.dot(newiDCT,fivePercent)
# reconstruction = np.dot(DCT_F_sort,newiDCT)
shaped = np.reshape(reconstruction, (53, 41))

plt.figure(4)
plt.imshow(shaped, cmap='gray')
plt.title(str(0.2 * 100) + "% image")


# for 40%
percent = round(0.4 * len(DCT_F_sort))
print('For ' + str(0.4) + ' there are ' + str(percent) + ' values to include')
threshold = DCT_F_sort[percent]
# testing = np.percentile(DCT_F_sort)
# fivePercent = np.zeros(pixels)
# fivePercent[0:percent] = DCT_F_sort[0:percent]
for j in range(0, len(DCT_F)):
    if abs(dctF40[j]) < threshold:
        dctF40[j] = 0
    else:
        dctF40[j] = dctF40[j]
reconstruction = np.dot(newiDCT, dctF40)
# reconstruction = np.dot(newiDCT,fivePercent)
# reconstruction = np.dot(DCT_F_sort,newiDCT)
shaped = np.reshape(reconstruction, (53, 41))

plt.figure(5)
plt.imshow(shaped, cmap='gray')
plt.title(str(0.4 * 100) + "% image")



# TASK 2 (Compressed Image Recovery) #################################################################################
# Your goal in this step is to recover the image F from limited random observations of its pixels using the prior
# knowledge that the DCT(F) is nearly sparse.


# 2a) Let M < N be an integer and construct a measurement matrix B of size MxN by randomly selecting M rows of the
# identity matrix I size NxN

# make a for-loop so it calculates this for three trials
fig, ax = plt.subplots(3, 3, figsize=(8,8))

trial = [1,2,3]
M_values = [0.2, 0.4, 0.6]
for j in range (0,len(M_values)):
    for i in range(0, len(trial)):
        # take random permutation of N
        # then take the first M entries
        N = pixels
        r = M_values[j]  # this will be what we iterate for 0.2, 0.4, and 0.6
        M = int(r * N)

        I = np.identity(pixels)  # NxN
        B_total = np.random.permutation(I)  # MxN
        B = B_total[0:M, :]

        # f_vals = our flattened image
        # y = B dot image_flat,
        y = np.dot(B, f)  # multiply by the flattened image
        # A = B dot iDCT
        A = np.dot(B, newiDCT)  # multipy by the inverse DCT to reconstruct

        # cvx optimization solving
        # x is CLEARLY the DCT vector of an image F* that hopefully resembles the original image
        x_l1 = cvx.Variable(N)

        # alt formulation
        objective_l1 = cvx.Minimize(cvx.norm(x_l1, 1))
        constraints_l1 = [A @ x_l1 == y]  # constaint in cvs = A @ x == y
        prob_l1 = cvx.Problem(objective_l1, constraints_l1)

        prob_l1.solve(verbose=True, solver='CVXOPT', max_iter=1000, reltol=1e-2, featol=1e-2)

        # result should be Nx1
        result = x_l1.value

        # then do the iDCT and reshape it back to the original image like we did in task 1
        d_reconstruction = np.dot(newiDCT, result)
        d_shaped = np.reshape(d_reconstruction, (53, 41))

        # plt.figure(int(i+10))
        ax[j,i].imshow(d_shaped, cmap='gray')
        ax[j,i].set_title("Trial " + str(i + 1) + ": M = " + str(M_values[j]))
        # plt.imshow(d_shaped, cmap='gray')
        # plt.title("Optimized image")












