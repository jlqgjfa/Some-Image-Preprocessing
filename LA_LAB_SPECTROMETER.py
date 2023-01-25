# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 20:38:38 2021

@author: jlqgj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
import matplotlib

##Importing and loading file 
file_1 = 'laser_1.txt'

laser_1 = np.loadtxt(file_1)

#Basics, Explore data
print(type(laser_1))
print(laser_1.shape) #(rows, columns)
print(laser_1.size)  #(total values, rows*columns)
# print(laser_1) #prints txt file

# Extracting Columns
laser_spectrum = laser_1[:,0]
print('laser spectrum ', laser_spectrum)
laser_intensity = laser_1[:,1]
print('laser intensity ', laser_intensity)

cn = {'fontname':'Courier New'}

fig, ax = plt.subplots(1)
ax.plot(laser_spectrum, laser_intensity, color='r', linewidth=.7)
plt.title('Laser', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity Ι %', **cn, fontsize=14)
plt.grid(True)
fig.show()

# vh_1 =[i for i,v in enumerate(laser_intensity>= 1.5) if v]
# print(vh_1)

##################################################################

file_2 = 'laser_1_hg.txt'

hg = np.loadtxt(file_2)

#Basics, Explore data
print(type(hg))
print(hg.shape) #(rows, columns)
print(hg.size)  #(total values, rows*columns)
# print(hg) #prints txt file

# Extracting Columns
hg_spectrum = hg[:,0]
print('Mercury spectrum ', hg_spectrum)
hg_intensity = hg[:,1]
print('Mercury Intensity ', hg_intensity)


fig2, ax2 = plt.subplots(1)
ax2.plot(hg_spectrum, hg_intensity, color='gray', linewidth=.7)
plt.title('Hg', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity Ι %', **cn, fontsize=14)
plt.grid(True)
fig2.show()

# vh_hg =[i for i,v in enumerate(hg_intensity>= 20) if v]
# print(vh_hg)

#################################################################

file_3 = 'laser_1_he.txt'

he = np.loadtxt(file_3)

#Basics, Explore data
print(type(he))
print(he.shape) #(rows, columns)
print(he.size)  #(total values, rows*columns)
# print(hg) #prints txt file

# Extracting Columns
he_spectrum = he[:,0]
print('He spectrum ', he_spectrum)
he_intensity = he[:,1]
print('He Intensity ', he_intensity)


fig3, ax3 = plt.subplots(1)
ax3.plot(he_spectrum, he_intensity, color='darkviolet', linewidth=.7)
plt.title('He', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity Ι %', **cn, fontsize=14)
plt.grid(True)
fig3.show()

#################################################################

file_4 = 'laser_1_ar.txt'

ar = np.loadtxt(file_4)

#Basics, Explore data
print(type(ar))
print(ar.shape) #(rows, columns)
print(ar.size)  #(total values, rows*columns)
# print(hg) #prints txt file

# Extracting Columns
ar_spectrum = ar[:,0]
print('Ar spectrum ', ar_spectrum)
ar_intensity = ar[:,1]
print('Ar Intensity ', ar_intensity)


fig4, ax4 = plt.subplots(1)
ax4.plot(ar_spectrum, ar_intensity, color='darkkhaki', linewidth=.7)
plt.title('Ar', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity Ι %', **cn, fontsize=14)
plt.grid(True)
fig4.show()

################################################################

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


n_samples_X = len(ar_spectrum)
ar_rs = ar_spectrum.reshape((n_samples_X,1)) #reshaping to (n,1)

X = ar_rs
y = ar_intensity

# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)

print('X shape',X.shape)
print('y shape',y.shape)
print('y data type', type(y))

svr_rbf = SVR(kernel='rbf', C=100, gamma=.001)
y_rbf = svr_rbf.fit(X, y).predict(X)

# s_v = input ('Spectrum Value = ')
# pred_intensity = svr_rbf.predict([[s_v]])
# print('Predicted Intensity = ', pred_intensity)



print('y shape',y_rbf.shape)
print(y_rbf)
fig5, ax5 = plt.subplots(1)
ax5.scatter(X, y, s=1.5, color='orange', label='Acquired data')
ax5.plot(X, y_rbf, color='blue', label='RBF model',linewidth=1)
plt.title('Prediction (Ar)', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity Ι %', **cn, fontsize=14)
plt.legend(loc="upper left")
plt.grid(True)
fig5.show()

################################################################

file_5 = 'laser_1_bb12v.txt'

bb12v = np.loadtxt(file_5)

#Basics, Explore data
print(type(bb12v))
print(bb12v.shape) #(rows, columns)
print(bb12v.size)  #(total values, rows*columns)
# print(hg) #prints txt file

# Extracting Columns
bb12v_spectrum = bb12v[:,0]
print('Ar spectrum ', bb12v_spectrum)
bb12v_intensity = bb12v[:,1]
print('Ar Intensity ', bb12v_intensity)


fig6, ax6 = plt.subplots(1)
ax6.plot(bb12v_spectrum, bb12v_intensity, color='darkkhaki', linewidth=.7)
plt.title('Halogen Lamp', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity I3/Ι %', **cn, fontsize=14)
plt.grid(True)
fig6.show()

#PREDICTION BB
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


n_samples_bb12v = len(bb12v_spectrum)
bb12v_rs = bb12v_spectrum.reshape((n_samples_bb12v,1)) #reshaping to (n,1)

X_bb12 = bb12v_rs
y_bb12 = bb12v_intensity

# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)

# print('X shape',X_bb12.shape)
# print('y shape',y_bb12.shape)
# print('y data type', type(y_bb12))

bb12_svr_rbf = SVR(kernel='rbf', C=10, gamma=.001)
bb12_y_rbf = bb12_svr_rbf.fit(X_bb12, y_bb12).predict(X_bb12)

# bb12s_v = input ('Spectrum Value = ')

# bb12pred_intensity = bb12_svr_rbf.predict([[bb12s_v]])

# print('Predicted Intensity = ', bb12pred_intensity)


fig7, ax7 = plt.subplots(1)
ax7.scatter(X_bb12, y_bb12, s=1.5, color='orange', label='Acquired data')
ax7.plot(X_bb12, bb12_y_rbf, color='blue', label='RBF model',linewidth=1)
plt.title('Prediction (Black Body 12v)', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity I3/Ι %', **cn, fontsize=14)
plt.legend(loc="upper left")
plt.grid(True)
fig7.show()

###############################################################
file_11 = 'laser_1_bb11v.txt'
file_10 = 'laser_1_bb10v.txt'
file_9 = 'laser_1_bb9v.txt'
file_8 = 'laser_1_bb8v.txt'
file_7 = 'laser_1_bb7v.txt'
file_6 = 'laser_1_bb6v.txt'

bb11v = np.loadtxt(file_11)
bb10v = np.loadtxt(file_10)
bb9v = np.loadtxt(file_9)
bb8v = np.loadtxt(file_8)
bb7v = np.loadtxt(file_7)
bb6v = np.loadtxt(file_6)


# Extracting Columns
bb11v_spectrum = bb11v[:,0]
bb11v_intensity = bb11v[:,1]

bb10v_spectrum = bb10v[:,0]
bb10v_intensity = bb10v[:,1]

bb9v_spectrum = bb9v[:,0]
bb9v_intensity = bb9v[:,1]

bb8v_spectrum = bb8v[:,0]
bb8v_intensity = bb8v[:,1]

bb7v_spectrum = bb7v[:,0]
bb7v_intensity = bb7v[:,1]

bb6v_spectrum = bb6v[:,0]
bb6v_intensity = bb6v[:,1]

#Fitting
n_samples_bb11v = len(bb11v_spectrum)
bb11v_rs = bb11v_spectrum.reshape((n_samples_bb11v,1)) #reshaping to (n,1)

X_bb11 = bb11v_rs
y_bb11 = bb11v_intensity

bb11_svr_rbf = SVR(kernel='rbf', C=10, gamma=.001)
bb11_y_rbf = bb11_svr_rbf.fit(X_bb11, y_bb11).predict(X_bb11)
################################################################
n_samples_bb10v = len(bb10v_spectrum)
bb10v_rs = bb10v_spectrum.reshape((n_samples_bb10v,1)) #reshaping to (n,1)

X_bb10 = bb10v_rs
y_bb10 = bb10v_intensity

bb10_svr_rbf = SVR(kernel='rbf', C=10, gamma=.001)
bb10_y_rbf = bb10_svr_rbf.fit(X_bb10, y_bb10).predict(X_bb10)
################################################################
n_samples_bb9v = len(bb9v_spectrum)
bb9v_rs = bb9v_spectrum.reshape((n_samples_bb9v,1)) #reshaping to (n,1)

X_bb9 = bb9v_rs
y_bb9 = bb9v_intensity

bb9_svr_rbf = SVR(kernel='rbf', C=10, gamma=.001)
bb9_y_rbf = bb9_svr_rbf.fit(X_bb9, y_bb9).predict(X_bb9)
###############################################################
n_samples_bb8v = len(bb8v_spectrum)
bb8v_rs = bb8v_spectrum.reshape((n_samples_bb8v,1)) #reshaping to (n,1)

X_bb8 = bb8v_rs
y_bb8 = bb8v_intensity

bb8_svr_rbf = SVR(kernel='rbf', C=10, gamma=.001)
bb8_y_rbf = bb8_svr_rbf.fit(X_bb8, y_bb8).predict(X_bb8)
#############################################################
n_samples_bb7v = len(bb7v_spectrum)
bb7v_rs = bb7v_spectrum.reshape((n_samples_bb7v,1)) #reshaping to (n,1)

X_bb7 = bb7v_rs
y_bb7 = bb7v_intensity

bb7_svr_rbf = SVR(kernel='rbf', C=10, gamma=.001)
bb7_y_rbf = bb7_svr_rbf.fit(X_bb7, y_bb7).predict(X_bb7)
##############################################################
n_samples_bb6v = len(bb6v_spectrum)
bb6v_rs = bb6v_spectrum.reshape((n_samples_bb6v,1)) #reshaping to (n,1)

X_bb6 = bb6v_rs
y_bb6 = bb6v_intensity

bb6_svr_rbf = SVR(kernel='rbf', C=10, gamma=.001)
bb6_y_rbf = bb6_svr_rbf.fit(X_bb6, y_bb6).predict(X_bb6)





# bb11s_v = input ('Spectrum Value = ')
# bb11pred_intensity = bb11_svr_rbf.predict([[bb11s_v]])
# print('Predicted Intensity = ', bb11pred_intensity)




fig8, ax8 = plt.subplots(1)
ax8.scatter(bb12v_spectrum, bb12v_intensity, color='silver', label='12V',s=.4)
ax8.scatter(bb11v_spectrum, bb11v_intensity, color='aqua', label='11V',s=.4)
ax8.scatter(bb10v_spectrum, bb10v_intensity, color='violet', label='10V',s=.4)
ax8.scatter(bb9v_spectrum, bb9v_intensity, color='gold', label='9V',s=.4)
ax8.scatter(bb8v_spectrum, bb8v_intensity, color='lime', label='8V',s=.4)
ax8.scatter(bb7v_spectrum, bb7v_intensity, color='black', label='7V',s=.4)
ax8.scatter(bb6v_spectrum, bb6v_intensity, color='pink', label='6V',s=.4)
ax8.plot(X_bb12, bb12_y_rbf, color='red', label='RBF model',linewidth=.7)
ax8.plot(X_bb11, bb11_y_rbf, color='red',linewidth=.7)
ax8.plot(X_bb10, bb10_y_rbf, color='red',linewidth=.7)
ax8.plot(X_bb9, bb9_y_rbf, color='red',linewidth=.7)
ax8.plot(X_bb8, bb8_y_rbf, color='red',linewidth=.7)
ax8.plot(X_bb7, bb7_y_rbf, color='red',linewidth=.7)
ax8.plot(X_bb6, bb6_y_rbf, color='red',linewidth=.7)
plt.title('Halogen Lamp V, svr (RBF)', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity I3/Ι %', **cn, fontsize=14)
plt.legend(loc="upper left", title='Temperature')
plt.grid(True)
fig8.show()




###################################################################################

file_12 = 'laser_1_bbi2.txt'

bbi2 = np.loadtxt(file_12)

# Extracting Columns
bbi2_spectrum = bbi2[:,0]
bbi2_intensity = bbi2[:,1]


n_samples_bbi2 = len(bbi2_spectrum)
bbi2_rs = bbi2_spectrum.reshape((n_samples_bbi2,1)) #reshaping to (n,1)

X_bbi2 = bbi2_rs
y_bbi2 = bbi2_intensity


bbi2_svr_rbf = SVR(kernel='rbf', C=10, gamma=.0001)
bbi2_y_rbf = bbi2_svr_rbf.fit(X_bbi2, y_bbi2).predict(X_bbi2)

# bb12s_v = input ('Spectrum Value = ')
# bb12pred_intensity = bb12_svr_rbf.predict([[bb12s_v]])
# print('Predicted Intensity = ', bb12pred_intensity)


fig9, ax9 = plt.subplots(1)
ax9.scatter(X_bbi2, y_bbi2, s=1, color='gold', label='Acquired data')
ax9.plot(X_bbi2, bbi2_y_rbf, color='blue', label='RBF model',linewidth=1)
plt.title('Prediction (Black Body i2)', **cn, fontsize=14)
plt.xlabel('Spectrum nm λ', **cn, fontsize=14)
plt.ylabel('Intensity I2 %', **cn, fontsize=14)
plt.legend(loc="upper right")
plt.grid(True)
fig9.show()
############################################################################
from numpy import log as ln
#findig max value

result = np.where(he_intensity == np.amax(he_intensity))
print(result)
index = [428]
print(index)


Temperature = lambda t: (1.986445857e-25)/((ln(((1.191042972e-16)/(t**5))+1))*(t)*(1.380649e-23))
temp = np.array([Temperature(bb11v_spectrum) for bb11v_spectrum in bb11v_spectrum])
print(temp)
