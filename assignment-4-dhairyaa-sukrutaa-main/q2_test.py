# -*- coding: utf-8 -*-
"""Q2_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R-h6mR7sy_h7Yi1dKE4xCb8nKprgkVDj
"""

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linear_regression import LinearRegressionCustom
from metrics import *
import time
np.random.seed(45)


# finding the optimal hyperparameters by individually varying the hyperparameters and finding the best combination of hyperparameters
N = 90
P = 10
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

# finding the optimal batch size
batch_sizes = [1, 4, 12, 18, 54]
epochs = [10, 20, 30, 40]
lrs = [0.1, 0.01, 0.001]
ridge_lambdas = [0.1, 0.01, 0.001]
lasso_lambdas = [0.1, 0.01, 0.001]
momentums = [0.9, 0.8, 0.7]
outFolds = 5
inFolds = 4
ind = 18
rmsefinal = 0
for i in range(outFolds):
    X_test = pd.DataFrame(X[i*ind:(i+1)*ind]).reset_index(drop=True)
    y_test = pd.Series(y[i*ind:(i+1)*ind]).reset_index(drop=True)
    X_mainTrain = pd.concat(
        (X[0:i*ind], X[(i+1)*ind:]), axis=0).reset_index(drop=True)
    y_mainTrain = pd.concat(
        (y[0:i*ind], y[(i+1)*ind:]), axis=0).reset_index(drop=True)
    optrmse = None
    final_batch_size = None
    for batch_size in batch_sizes:
        avgrmse = 0
        for j in range(inFolds):
            X_actualTrain = pd.concat(
                (X_mainTrain[0:j*ind], X_mainTrain[(j+1)*ind:]), axis=0).reset_index(drop=True)
            y_actualTrain = pd.concat(
                (y_mainTrain[0:j*ind], y_mainTrain[(j+1)*ind:]), axis=0).reset_index(drop=True)
            X_validation = pd.DataFrame(
                X_mainTrain[j*ind:(j+1)*ind]).reset_index(drop=True)
            y_validation = pd.Series(
                y_mainTrain[j*ind:(j+1)*ind]).reset_index(drop=True)
            LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                        ridge_lambda=0.1, batch_size=batch_size, lr=0.01, epochs=30, momentum=0.9)
            LR.fit_gradient_descent(
                X_actualTrain, y_actualTrain, 'manual', 'l2')
            y_hat = LR.predict_kfold(X_validation)

            avgrmse += rmse(y_hat, y_validation)
        avgrmse /= inFolds
        if optrmse is None or avgrmse < optrmse:
            optrmse = avgrmse
            print(optrmse, batch_size)
            final_batch_size = batch_size
    finalLR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1, ridge_lambda=0.1,
                                     batch_size=final_batch_size, lr=0.01, epochs=30, momentum=0.9)
    finalLR.fit_gradient_descent(X_mainTrain, y_mainTrain, 'manual', 'l2')
    y_hat = finalLR.predict_kfold(X_test)
    rmsefinal += rmse(y_hat, y_test)
rmsefinal /= outFolds
print('Optimal batch size =', final_batch_size)
print('RMSE: ', rmsefinal)
# did the above for all hyperparameters and found the best combination of hyperparameters
# batch_size, epochs, lr, ridge, lasso, momentum
good_hyperparams = [4, 30, 0.1, 0.01, 0.1, 0.9]
# Testing all the test cases given in the question using default hyperparameters
N = 90
P = 10
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
exp = 5

# fit_gradient_descent using the manually computed gradients for each of unregularized mse_loss, and mse_loss with ridge regularization
times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'manual', 'unregularized')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('For gradient descent with unregularized mse_loss : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))

times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'manual', 'l2')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('For gradient descent with ridge regularization : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))


# fit_gradient_descent using the JAX gradients for each of unregularized mse_loss, mse_loss with LASSO regularization and mse_loss with ridge regression

times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'jax', 'unregularized')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('For gradient descent with unregularized mse loss : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))

times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'jax', 'l1')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('For gradient descent with lasso regularization : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))

times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'jax', 'l2')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('For gradient descent with ridge regularization : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))


# fit_gradient_descent for running SGD on mse_loss with ridge regularization
times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=1, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'manual', 'l2')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('SGD with ridge regularization(manual) : \n')
print(' Number of batches =', N)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))

times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=1, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'jax', 'l2')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('SGD with ridge regularization(jax): \n')
print(' Number of batches =', N)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))

# fit_gradient_descent for running minibatch SGD on mse_loss with ridge regularization
times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'manual', 'l2')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('SGD minibatch with ridge regularization : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))

times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_gradient_descent(X, y, 'jax', 'l2')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('SGD minibatch with ridge regularization(jax) : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))

# fit_SGD_with_momentum for running SGD on mse_loss with ridge regularization
times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_SGD_with_momentum(X, y, 'l2')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('SGD MOMENTUM with ridge regularization(manual) : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))

times = []
for i in range(exp):
    begin = time.time()
    LR = LinearRegressionCustom(fit_intercept=True, lasso_lambda=0.1,
                                ridge_lambda=0.1, batch_size=18, lr=0.01, epochs=30, momentum=0.9)
    LR.fit_SGD_with_momentum(X, y, 'l2')
    y_hat = LR.predict(X)
    end = time.time()
    times.append(end-begin)

print('SGD MOMENTUM with ridge regularization(jax) : \n')
print(' Number of batches =', N//18)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print('Time taken: ', np.mean(times))