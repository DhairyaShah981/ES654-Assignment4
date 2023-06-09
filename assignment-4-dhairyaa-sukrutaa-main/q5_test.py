# -*- coding: utf-8 -*-
"""Q5_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19fLBYgrxvEcH2qfm0URlFaLuEUyLMibB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linear_regression import LinearRegressionCustom
import os.path
from os import path

if not path.exists('Plots/Question5/'):
    os.makedirs('Plots/Question5/')

# TODO : Write here

def make_data(degrees, include_bias = True, N = 17):
    li_master = []
    for n in range(10,N,1):
        li = []
        x = np.array([i*np.pi/180 for i in range(60,(2*n)+60+1,2)])
        y = 3*x + 8 + np.random.normal(0,3,len(x)) 
        for deg in degrees:
            go = PolynomialFeatures(deg,include_bias)
            li_1 = []
            for i in range(len(x)):
                j = np.array([x[i]])
                j = go.transform(j)
                li_1.append(j)
            X = np.array(li_1)
            lin_reg = LinearRegressionCustom(fit_intercept=include_bias)
            lin_reg.fit_sklearn_LR(X,y)
            the_tas = lin_reg.coef_
            norm = np.linalg.norm(np.array(the_tas))
            li.append(norm)
        li_master.append(li)
    return(li_master)

def make_plots(degrees, include_bias =True, N = 13):
    li = make_data(degrees)
    count = 0
    for x in li:
        na = "N = " + str(1*count+10)
        plt.plot(degrees,x, label = na)
        plt.xlabel("Degree of fitted polynmial")
        plt.ylabel("Magnitude of theta")
        plt.yscale("log")
        var = "./Plots/Question5/Plot_5__" + ".png"
        plt.legend()
        plt.savefig(var, dpi=400)
        
        count += 1


make_plots([1,3,5,6,7,9])







