# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 23:14:25 2022

@author: jlqgj
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import io, color
from sklearn.preprocessing import MinMaxScaler
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import numpy as np
from skimage.measure import moments
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.color import rgb2gray,rgba2rgb
import cv2 as cv
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from scipy.stats import moment


#%%
#Area moment of inertia, it is not a unique property of a cross section.
#It quantifies the resistance to bending in a particular axis, and so its value changes
#depending on where we place this reference axis. One can aproximates the moment of inertia 
#of a cross section by splitting it into small elements... dA(small part of area)     

input_image = io.imread('C:/Users/0100_x1y1.png')

image = input_image
#Plotting Original image
plt.imshow(image, origin="lower", cmap="gray")
plt.xlabel("x pixels"); plt.ylabel("y pixels")
plt.grid()
plt.colorbar(label="Norm Intensity 0-255")

#converting to grayscale
grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#converting to binary image
ret,thresh1 = cv.threshold(grayscale,110,255,cv.THRESH_BINARY)

#counting num of total white/black pixels
num_white_pixels = np.sum(thresh1 == 255)
print('count white pixels ', num_white_pixels)
num_black_pixels = np.sum(thresh1 == 0)
print('count black pixels ', num_black_pixels)


#Plotting binary image
cn = {'fontname':'Courier New'}

fig, ax = plt.subplots(1)
ax.imshow(thresh1)
plt.title('Binary Image', **cn, fontsize=14)
plt.xlabel('Pixels 1024', **cn, fontsize=14)
plt.ylabel('Pixels 1024', **cn, fontsize=14)
plt.grid(True)
fig.show()


# find contours in the binary image
contours, hierarchy = cv.findContours(thresh1,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

var_r = [None]*len(contours)
ell = [None]*len(contours)

for i,c in enumerate(contours):
   # calculate moments for each contour
   M = cv.moments(c)
   print('Moments ',M)
   # calculating hu moments
   huMoments = cv.HuMoments(M)
   # calculate x,y coordinate of center
   #m00 being the area
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   print('cx',cX)
   print('cy',cY)
   print('hu moments', huMoments)
   cv.circle(image, (cX, cY), 5, (255, 255, 255)[-2], -5)
   cv.putText(image, "c="+str(cX)+','+str(cY), (cX - 100, cY - 100),cv.FONT_ITALIC,0.7, (255, 255, 255))
   var_r[i] = cv.boundingRect(contours[i]) #storing contours values
   ellipse = cv.fitEllipse(c)
   ell[i] = cv.fitEllipse(contours[i]) 
   print('ellipse ',ellipse)
   cv.ellipse(image,ellipse,(0,255,0),2)
   
   
# print(contours)
print('Raw contours x1,y1,x2,y2 ',var_r)
print('ellipse ', ell)
rect_array = np.array(var_r)
print(i)
print('t ',ell[1][0][0])


x1=[]
y2=[]
x2=[]
y2=[]
c_y=[]
c_x=[]
mean_pixels=[]
for p in range(i+1):
    x1 = (rect_array[p][0])
    y1 = (rect_array[p][1])
    x2 = (rect_array[p][0])+(rect_array[p][2])
    y2 = (rect_array[p][1])+(rect_array[p][3])
    c_y = (rect_array[p][1])+(((rect_array[p][3])/2))
    c_x = (rect_array[p][0])+(((rect_array[p][2])/2))
    cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
    mean_pixels = np.round(np.mean((image[y1:y2, x1:x2])),decimals=2)
    cv.putText(image, "I= "+str(mean_pixels), (x1 - 5, y1 - 5),cv.FONT_ITALIC,0.5, (255, 255, 255))
    print(str(p)+' mean I ', mean_pixels)
    # Count number of white pixels:
    whitePixels = cv.countNonZero(thresh1[y1:y2, x1:x2])
    cv.putText(image, "No.WP: "+str(whitePixels), (x1 - 19, y1 - 19),cv.FONT_ITALIC,0.5, (255, 255, 255))
    print(str(p)+" No.White Pixels: "+' '+str(whitePixels))
    print(str(p)+' cx: '+str(c_x)+' '+'cy: '+str(c_y))

   
#Centroids
cn = {'fontname':'Courier New'}
fig2, ax2 = plt.subplots(1)
ax2.imshow(image)
plt.title('Centroid, Segmentation, Intensity 0-255 (mean)', **cn, fontsize=14)
plt.xlabel('Pixels 1024', **cn, fontsize=14)
plt.ylabel('Pixels 1024', **cn, fontsize=14)
plt.grid(False)
fig2.show()   

# #Histogram of colors grayscale
# fig4, ax4 = plt.subplots(1)
# plt.hist(grayscale.ravel(),256,[0,256]); plt.show()
# plt.title('Histogram (grayscale)', **cn, fontsize=14)
# #ax4.set_xlim([0, 260])
# ax4.set_ylim([0, 1e5])
# plt.xlabel('0-255 (grayscale)', **cn, fontsize=14)
# plt.ylabel('Count', **cn, fontsize=14)
# plt.grid(True)
# fig4.show() 

