import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from linearRegression.linear_regression import LinearRegressionCustom
import matplotlib.pyplot as plt
import os.path
from os import path
np.random.seed(45)

if not path.exists('Plots/Question6/'):
    os.makedirs('Plots/Question6/')


rmse_values_norm = []
rmse_values_unnorm = []
for ri in range(2):
    li_norm = []
    li_unnorm = []
    for n in range(100,120):
        means = np.array([4, 3, 5, 3, 5, 4, 12, 23, 9, 0])
        variances = np.array([52, 25, 2, 1, 1, 1, 2, 0.1, 0.1, 0.1])
        X = pd.DataFrame(np.random.normal(means, variances, size=(n, 10)))
        y = 3*X[0] + 4*X[1] + 8 +  np.random.normal(0, 3, len(X))
        if ri == 1:
            means = np.array([40, 3, 59, 37, 56, 45, 12, 23, 59, 70])
            variances = np.array([52, 25, 2, 1, 1, 1, 2, 0.1, 0.1, 0.1])
            X = pd.DataFrame(np.random.normal(means, variances, size=(n, 10)))
            y = 3*X[0] + 4*X[1] + 8 +  np.random.normal(0, 3, len(X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr = LinearRegressionCustom()
        lr.fit_sklearn_LR(X_train, y_train)
        lr_scaled = LinearRegressionCustom(fit_intercept=False)
        lr_scaled.fit_sklearn_LR(X_train_scaled, y_train)
        y_pred = lr.predict_kfold(X_test)
        y_pred_scaled = lr_scaled.predict_kfold(X_test_scaled)
        mse_unnorm = mse = np.mean((y_pred- y_test)**2)
        mse_norm = np.mean((y_pred_scaled- y_test)**2)
        rmse_unnorm = np.sqrt(mse_unnorm)
        rmse_norm = np.sqrt(mse_norm)
        li_unnorm.append(rmse_unnorm)
        li_norm.append(rmse_norm)
    rmse_values_unnorm.append(li_unnorm)
    rmse_values_norm.append(li_norm)

n = range(100, 120)

for x in range(2):
    plt.clf()
    plt.plot(n, rmse_values_unnorm[x], label="RMSE_unnorm")
    plt.plot(n, rmse_values_norm[x], label="RMSE_norm")
    plt.xlabel("n")
    plt.ylabel("RMSE")
    plt.title("Comparing the RMSE values")
    plt.legend()
    var = "./Plots/Question6/Plot_6_highmean_" + str(x) + ".png"
    plt.savefig(var, dpi=400)
