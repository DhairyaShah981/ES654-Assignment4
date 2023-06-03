# -*- coding: utf-8 -*-
"""Q3_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dWm-EWFDEwI15WbGIT0_bUIrxolF7Z1G
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linear_regression import LinearRegressionCustom
from os import path
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

# dataset and fitting
N = 60
x = np.array([i*np.pi/180 for i in range(60, 300, 2)])
np.random.seed(10)
y = 3*x + 8 + np.random.normal(0, 3, len(x))

y = pd.Series(y)
LR = LinearRegressionCustom(
    fit_intercept=True, batch_size=40, lr=0.01, epochs=10)
LR.fit_gradient_descent(pd.DataFrame(x), y, 'jax', 'l1')


# Line fit animation
fig, ax = plt.subplots()


def update_graph(t):
    ax.clear()
    LR.plot_line_fit(pd.Series(x), y, [coef[0] for coef in LR.all_coef][t], [
                     coef[1] for coef in LR.all_coef][t], ax)
    ax.set_xlim(0.9, 5.5)
    ax.set_ylim(0, 34)


images = []
for t in range(0, 30):
    update_graph(t)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

imageio.mimsave('./Plots/lineFit.gif', images, fps=5)


# Contour plot animation
fig, ax = plt.subplots()


def update_graph(t):
    ax.plot([coef[0] for coef in LR.all_coef][t], [coef[1]
            for coef in LR.all_coef][t], 'ro')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    if t < 29:
        current = np.array([LR.all_coef[t][0], LR.all_coef[t][1]])
        next_ = np.array([LR.all_coef[t+1][0], LR.all_coef[t+1][1]])
        ax.annotate("", xy=current, xytext=next_, arrowprops=dict(
            arrowstyle="<-", color='black', lw=1))
    return ax


def new_update():
    theta_0_range = np.linspace(-6, 6, 400)
    theta_1_range = np.linspace(-6, 6, 400)
    rss = np.zeros((len(theta_0_range), len(theta_1_range)))
    for i, theta_0_val in enumerate(theta_0_range):
        for j, theta_1_val in enumerate(theta_1_range):
            LR.coef_ = np.array([theta_0_val, theta_1_val])
            y_pred = LR.predict(x)
            rss[i, j] = np.sum((y_pred - y)**2)

    # Create the plot
    theta_0_mesh, theta_1_mesh = np.meshgrid(theta_0_range, theta_1_range)

    # Define the levels to use for the contours
    max_rss = np.max(rss)
    levels_outer = np.linspace(0, max_rss, 20)
    levels_inner = np.linspace(0, max_rss, 100)

    # Create the contours with denser levels near the center
    contours_outer = ax.contour(
        theta_0_mesh, theta_1_mesh, rss.T, levels=levels_outer, cmap='viridis')
    contours_inner = ax.contour(
        theta_0_mesh, theta_1_mesh, rss.T, levels=levels_inner, cmap='viridis')

    ax.clabel(contours_outer, inline=True, fontsize=8)
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_title('Residual Sum of Squares Contour Plot')


images = []
new_update()
for t in range(30):
    update_graph(t)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

imageio.mimsave('./Plots/contourPlots.gif', images, fps=5)

# Surface plot animation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rss = LR.plot_surface_init(x, y, ax)


def update_graph(t):
    LR.coef_ = np.array([[coef[0] for coef in LR.all_coef][t], [
                        coef[1] for coef in LR.all_coef][t]])
    y_pred = LR.predict(X)
    rss = np.sum((y_pred - y)**2)
    ax.scatter([coef[0] for coef in LR.all_coef][t], [coef[1]
               for coef in LR.all_coef][t], rss, color='black')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    return ax


images = []
for t in range(30):
    update_graph(t)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

imageio.mimsave('./Plots/surfacePlots.gif', images, fps=5)