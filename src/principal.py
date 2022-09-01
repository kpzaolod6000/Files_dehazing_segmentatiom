# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 18:16:27 2022

@author: dox94
"""
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import cv2


# Import image
img = plt.imread("bloodcell.jpg")

# Create float
bgr = img.astype(float)/255.

# Extract channels
with np.errstate(invalid='ignore', divide='ignore'):
	K = 1 - np.max(bgr, axis=2)
	C = (1-bgr[...,2] - K)/(1-K)
	M = (1-bgr[...,1] - K)/(1-K)
	Y = (1-bgr[...,0] - K)/(1-K)

# Convert the input BGR image to CMYK colorspace
CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

# Split CMYK channels
Y, M, C, K = cv2.split(CMYK)

np.isfinite(C).all()
np.isfinite(M).all()
np.isfinite(K).all()
np.isfinite(Y).all()

# Save channels
cv2.imwrite('C.jpg', C)
cv2.imwrite('M.jpg', M)
cv2.imwrite('Y.jpg', Y)
cv2.imwrite('K.jpg', K)


#EqualizationHistogram
equ = cv2.equalizeHist(K)

# stacking images side-by-side
#res = np.hstack((Y, equ))
#print(res)
#plt.imshow(equ,cmap='gray')
###


#Linear Contrast
linear = (Y-np.amin(Y))*((255-0)/(np.amax(Y)-np.amin(Y)))
#plt.imshow(equ,cmap='gray')
ei = 2*linear + equ
#plt.imshow(ei)
####
#Filtro minimo
size = (3, 3)
shape = cv2.MORPH_RECT
kernel = cv2.getStructuringElement(shape, size)
imgResult = cv2.erode(ei, kernel)
imgResult = cv2.erode(imgResult, kernel)
imgResult = cv2.erode(imgResult, kernel)
plt.imshow(imgResult,cmap="gray")

####

#Otsu threesholding
