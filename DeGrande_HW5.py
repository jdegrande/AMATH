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

import scipy.fftpack as spfft # for discrete cosine transform


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

fig, ax = plt.subplots(1, 2, figsize=(20,10))
ax[0].imshow(img_og, cmap = 'gray')
ax[0].set_title("Original image")

print("Original size: ", img_og.shape)

# rescale the image using the code from the helper code
# resize image
img = ski.transform.rescale( img_og, 0.18, anti_aliasing=False)

print("Rescaled size: ", img.shape)

ax[1].imshow(img, cmap='gray')
ax[1].set_title("Rescaled image")


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

  Dx = spfft.dct(np.eye(Nx), axis =0, norm='ortho')
  Dy = spfft.dct(np.eye(Ny), axis = 0, norm='ortho')
  D = np.kron(Dy, Dx)

  return D

# construct inverse DCT matrix
def construct_iDCT_Mat(Nx, Ny ):
  # input : Nx number of columns of image
  #         Ny number of rows of image
  # output: iD iDCT matrix mapping DCT(image).flatten() to image.flatten()

  Dx = spfft.idct(np.eye(Nx), axis =0, norm='ortho')
  Dy = spfft.idct(np.eye(Ny), axis = 0, norm = 'ortho')
  D = np.kron(Dy, Dx)

  return D


newDCT = construct_DCT_Mat(numCols,numRows)
newiDCT = construct_iDCT_Mat(numCols,numRows)

f = np.ndarray.flatten(img)
index = np.arange(0,pixels)

DCT_F = np.matmul(newDCT, f)

# 1b) plot DCT(f) and investigate the compressibility. Do you see a lot of large coefficients??
plt.figure(3)
plt.scatter(index,DCT_F)
plt.title('DCT(F)')
plt.xlabel('index')
plt.ylabel('DCT(F)')


# 1c) reconstruct and plot the image after thresholding its DCT to keep the top 5,10,20, and 40 percent of DCT
# coefficients

DCT_F_copy = np.copy(DCT_F)
DCT_F_sort = np.abs(DCT_F_copy[np.argsort(-DCT_F_copy)])

percent = round(0.4 * len(DCT_F))
threshold = DCT_F_sort[percent]
#testing = np.percentile(DCT_F_sort)

fivePercent = np.zeros(pixels)
fivePercent[0:percent] = DCT_F[0:percent]

# for i in range(0,len(DCT_F_copy)):
#   if DCT_F_sort[i] <= threshold:
#     DCT_F_sort[i] = 0
#   else:
#     DCT_F_sort[i] = DCT_F_sort[i]

reconstruction = np.dot(fivePercent,newiDCT)
#reconstruction = np.dot(DCT_F_sort,newiDCT)
shaped = np.reshape(reconstruction, (53,41))

plt.figure(4)
plt.imshow(shaped, cmap='gray')
plt.title("Reconstructed 5%")