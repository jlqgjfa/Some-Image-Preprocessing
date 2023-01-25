# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:50:59 2022

@author: jlqgj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
import matplotlib
#Linear regression
import pandas as pd
from sklearn import linear_model, utils
import re

#reading txt files
#Laser Files
laser_15 = "C:/Users/jlqgj/programs/FOTM/laser/15.txt"
laser_20 = "C:/Users/jlqgj/programs/FOTM/laser/20.txt"
laser_25 = "C:/Users/jlqgj/programs/FOTM/laser/25.txt"
laser_30 = "C:/Users/jlqgj/programs/FOTM/laser/30.txt"
laser_35 = "C:/Users/jlqgj/programs/FOTM/laser/35.txt"

#LED Files
led_15 = "C:/Users/jlqgj/programs/FOTM/LED/15.txt"
led_20 = "C:/Users/jlqgj/programs/FOTM/LED/20.txt"
led_25 = "C:/Users/jlqgj/programs/FOTM/LED/25.txt"
led_30 = "C:/Users/jlqgj/programs/FOTM/LED/30.txt"
led_35 = "C:/Users/jlqgj/programs/FOTM/LED/35.txt"


tx1 = np.loadtxt(laser_15)
th = tx1[:,[0,1]]

print(th[None][0])

# for i in range(len(tx1[:,1])):
#     a =th[i][0]
#     b =th[i][1]
#     print(b)
#     if b>0.0001 and b<0.0002:
#         x = a
#         y = b
        


    
    
    
'''character	color 
‘b’	blue
‘g’	green
‘r’	red
‘c’	cyan
‘m’	magenta
‘y’	yellow
‘k’	black
‘w’	white '''

th15=[]
th20=[]
th25=[]
th30=[]
th35=[]


def findvalue_linear (textfile,lowlimit,highlimit,array1):
    txf = np.loadtxt(textfile)
    ext_2col = txf[:,[0,1]]
    array1=[]
    for i in range(len(txf[:,1])):
        x =ext_2col[i][0]
        y =ext_2col[i][1]
        
        if y>lowlimit and y<highlimit:
           print(i) 
           x1 = x
           array1.append(x1)
           y1 = y
           array1.append(y1)
           print(array1)
    return array1
           

thlaser15 = findvalue_linear(laser_15, 0.000035, 0.00004,th15)
thlaser20 = findvalue_linear(laser_20, 0.000044, 0.000046,th20)
thlaser25 = findvalue_linear(laser_25, 0.00003, 0.000033,th25)
thlaser30 = findvalue_linear(laser_30, 0.00005, 0.000052,th30)
thlaser35 = findvalue_linear(laser_35, 0.000024, 0.000025,th35)


def plot_and_read_currentvspower25 (txtfile, txtfile2, temperature, type1,label1,label2):
    txt_file = np.loadtxt(txtfile)
    txt_file2 = np.loadtxt(txtfile2)
    # Extracting Columns
    current = txt_file[:,0]
    power = txt_file[:,1]
    current1 = txt_file2[:,0]
    power1 = txt_file2[:,1]
    cn = {'fontname':'Courier New'}

    fig, ax = plt.subplots(1)
    ax.plot(current, power, color='r',linewidth=.7,label=label1+' ºC')
    ax.plot(current1, power1, color='c',linewidth=.7,label=label2+' ºC')
    plt.title(type1 + " " + str(temperature)+ " " +" ºC", **cn, fontsize=14)
    plt.xlabel('Current mA', **cn, fontsize=14)
    plt.ylabel('Power W ', **cn, fontsize=14)
    plt.legend(loc="upper left", title="Temperature")
    plt.grid(True)
    fig.show()
    
plot_and_read_currentvspower25(laser_25, led_25, 25, 'Laser & LED', 'Laser @25', 'LED @25')
# plot_and_read_currentvspower(laser_20, 20, 'r','Laser') 
# plot_and_read_currentvspower(laser_30, 30, 'b','Laser')

# plot_and_read_currentvspower(led_20, 20, 'g','LED') 
# plot_and_read_currentvspower(led_30, 30, 'm','LED')


def plot_allvalues_currentvspower (type1,txtfile1,txtfile2,txtfile3,txtfile4,txtfile5,label1,label2,label3,label4,label5):
    txt_file1 = np.loadtxt(txtfile1)
    txt_file2 = np.loadtxt(txtfile2)
    txt_file3 = np.loadtxt(txtfile3)
    txt_file4 = np.loadtxt(txtfile4)
    txt_file5 = np.loadtxt(txtfile5)
    
    # Extracting Columns
    current1 = txt_file1[:,0]
    power1 = txt_file1[:,1]
    current2 = txt_file2[:,0]
    power2 = txt_file2[:,1]
    current3 = txt_file3[:,0]
    power3 = txt_file3[:,1]
    current4 = txt_file4[:,0]
    power4 = txt_file4[:,1]
    current5 = txt_file5[:,0]
    power5 = txt_file5[:,1]
    
    
    cn = {'fontname':'Courier New'}
    fig1, ax1 = plt.subplots(1)
    ax1.plot(current1, power1, color='r',linewidth=.7,label=label1+' ºC')
    ax1.plot(current2, power2, color='c',linewidth=.7,label=label2+' ºC')
    ax1.plot(current3, power3, color='m',linewidth=.7,label=label3+' ºC')
    ax1.plot(current4, power4, color='b',linewidth=.7,label=label4+' ºC')
    ax1.plot(current5, power5, color='g',linewidth=.7,label=label5+' ºC')
    plt.title(type1, **cn, fontsize=14)
    plt.xlabel('Current mA ', **cn, fontsize=14)
    plt.ylabel('Power W ', **cn, fontsize=14)
    plt.legend(loc="upper left", title="Temperature")
    plt.grid(True)
    fig1.show()

plot_allvalues_currentvspower('Laser', laser_15, laser_20,laser_25,laser_30,laser_35,'15','20','25','30','35')
plot_allvalues_currentvspower('LED', led_15, led_20,led_25,led_30,led_35,'15','20','25','30','35')

def plot_allvalues_currentvsvoltage (type1,txtfile1,txtfile2,txtfile3,txtfile4,txtfile5,label1,label2,label3,label4,label5):
    txt_file1 = np.loadtxt(txtfile1)
    txt_file2 = np.loadtxt(txtfile2)
    txt_file3 = np.loadtxt(txtfile3)
    txt_file4 = np.loadtxt(txtfile4)
    txt_file5 = np.loadtxt(txtfile5)
    
    # Extracting Columns
    current1 = txt_file1[:,0]
    power1 = txt_file1[:,3]
    current2 = txt_file2[:,0]
    power2 = txt_file2[:,3]
    current3 = txt_file3[:,0]
    power3 = txt_file3[:,3]
    current4 = txt_file4[:,0]
    power4 = txt_file4[:,3]
    current5 = txt_file5[:,0]
    power5 = txt_file5[:,3]
    
    
    cn = {'fontname':'Courier New'}
    fig2, ax2 = plt.subplots(1)
    ax2.plot(current1, power1, color='r',linewidth=.7,label=label1+' ºC')
    ax2.plot(current2, power2, color='c',linewidth=.7,label=label2+' ºC')
    ax2.plot(current3, power3, color='m',linewidth=.7,label=label3+' ºC')
    ax2.plot(current4, power4, color='b',linewidth=.7,label=label4+' ºC')
    ax2.plot(current5, power5, color='g',linewidth=.7,label=label5+' ºC')
    plt.title(type1, **cn, fontsize=14)
    plt.xlabel('Current mA ', **cn, fontsize=14)
    plt.ylabel('Forward Voltage V ', **cn, fontsize=14)
    plt.legend(loc="upper left", title="Temperature")
    plt.grid(True)
    fig2.show()

plot_allvalues_currentvsvoltage('Laser', laser_15, laser_20, laser_25, laser_30, laser_35, '15', '20', '25', '30', '35')
plot_allvalues_currentvsvoltage('LED', led_15, led_20,led_25,led_30,led_35,'15','20','25','30','35')
    

    
def extracting_fitting(txtfile):
        txf = np.loadtxt(txtfile)
        ext_2col = txf[:,[0,3]]
        array1=ext_2col[44:,0]
        array2=ext_2col[44:,1]
        arrayx=[]
        arrayy=[]
        for j in range((len(txf[:,0]))-44):
            x =ext_2col[j][0]
            y =ext_2col[j][1]
            # print(j) 
            arrayx.append(x)
            arrayy.append(y)
            #print(array1)
        return array1, array2

x25, y25 = extracting_fitting(laser_25)




def extracting_fitting_led(txtfile):
        txf = np.loadtxt(txtfile)
        ext_2col = txf[:,[0,3]]
        array1=ext_2col[3:,0]
        array2=ext_2col[3:,1]
        arrayx=[]
        arrayy=[]
        for j in range((len(txf[:,0]))-2):
            x =ext_2col[j][0]
            y =ext_2col[j][1]
            # print(j) 
            arrayx.append(x)
            arrayy.append(y)
            #print(array1)
        return array1, array2

xled25, yled25 = extracting_fitting_led(led_25)
    



def linear_regression (x,y,value_to_predict):
    #Creating a dictionary
    fringes = {'Injection Current':x,'Voltage':y}
    df = pd.DataFrame(fringes, columns=['Injection Current','Voltage'])


    X_current = df[['Injection Current']]
    y_voltage = df[['Voltage']]
 
    X, y = utils.check_X_y(X_current, y_voltage, accept_sparse=True, dtype= list)

    fitting = linear_model.LinearRegression()
    fitting.fit(X, y)

    # value_to_predict = input ('Angle (º) = ')

    prediction = fitting.predict([[value_to_predict]])
    
    return prediction
  
Fitted_result = linear_regression(x25, y25, 37.5)
Fitted_result_led = linear_regression(xled25, yled25, 44)
# c = Fitted_result/0.0375
# cc = str(c)
# print('Series Resistor = ', cc, sep='')



prediction = []
for k in range(57):
    current = x25[k]
    g=linear_regression(x25, y25, current)
    # print(g)
    prediction.append(g)


# for removing the brackets flatten()
new_prediction = np.array(prediction).flatten()
p = new_prediction[31]
c = round(p/0.0375,2)



prediction_led = []
for l in range(38):
    current_led = xled25[l]
    gg=linear_regression(xled25, yled25, current_led)
    # print(g)
    prediction_led.append(gg)


# for removing the brackets flatten()
new_prediction_led = np.array(prediction_led).flatten()
p_led = new_prediction_led[17]
c_led = round(p/0.040,2)




def plot_and_read_currentvsvoltage (x, y, temperature, type1,label1):
    
    # Extracting Columns
    current = x
    voltage = y

    cn = {'fontname':'Courier New'}

    fig, ax = plt.subplots(1)
    ax.plot(current, voltage, color='r',linewidth=.7,label=label1)
    plt.title(type1 + " " + str(temperature)+ " " +" ºC", **cn, fontsize=14)
    plt.xlabel('Current mA', **cn, fontsize=14)
    plt.ylabel('Voltage V ', **cn, fontsize=14)
    #plt.legend(loc="upper left", title="Temperature & Series Resistor")
    plt.grid(True)
    fig.show() 
    
plot_and_read_currentvsvoltage(x25, new_prediction, 25, 'Fitting Linear Model (Laser)', '25'+' ºC ')#+str(c)+' Ω')
plot_and_read_currentvsvoltage(xled25,new_prediction_led, 25, 'Fitting Linear Model (LED)', '25'+' ºC ')


ith = [18.0,20.0,22.0,24.5,27.0]
tth = [15,20,25,30,35]

def plot_temp_voltage (x,y):
    
    i = y
    t = x

    cn = {'fontname':'Courier New'}

    fig, ax = plt.subplots(1)
    ax.scatter(i, t, color='r',s=20,label='Thresholds Values')
    plt.title('Thresholds Values', **cn, fontsize=14)
    plt.xlabel('Temperature ºC', **cn, fontsize=14)
    plt.ylabel('Ith Current mA', **cn, fontsize=14)
    plt.legend(loc="upper left", title="Ith vs Temp")
    plt.grid(True)
    fig.show()

plot_temp_voltage(ith, tth)
    




    