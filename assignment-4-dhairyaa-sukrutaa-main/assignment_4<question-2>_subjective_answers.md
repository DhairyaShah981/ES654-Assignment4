# Outputs for Gradient Descent

### K-Fold Nested Cross Validation Outputs 
We know that, in Batch Gradient Descent, Gradient Type can be jax or manual, and Penalty Type can be l1, l2, or unregularized. That's why every output is a dataframe with
first two columns as Gradient Type and Penalty Type, and we are trying all 6 combinations of them. <br> <br>
Also, in Batch Gradient Descent there are 5 hyperparameters which we need to tune using nested cross validation. Although when we tried doing the same for all 5 parameters
simulataneously, the program had 7 nested for loops which was very hard to run on Colab. The other option was to pick random combinations of all hyperparameters, and then find
the optimum set of hyperparameters.
<br><br>
Instead, we tried to see the trend in all hyperparameters individually which was fast, and easy to analyse. Following are the outputs for each hyperparameter tuning.
#### Tuning Batch Size
![image](https://user-images.githubusercontent.com/76472249/225251038-fbadad87-4ee3-41ad-99c5-f07c75e3b899.png)

#### Tuning Learning Rate
![image](https://user-images.githubusercontent.com/76472249/225251099-01a9d6ec-49bf-4474-829d-b11de2e08d0d.png)

#### Tuning Epochs
![image](https://user-images.githubusercontent.com/76472249/225251166-bf85c206-4ccc-4356-982f-b5f256a332cd.png)

#### Tuning Lasso Lambda
![image](https://user-images.githubusercontent.com/76472249/225251238-1b3ca975-23ca-45db-abc7-aeb476a92c45.png)

#### Tuning Ridge Lambda
![image](https://user-images.githubusercontent.com/76472249/225251307-e9e5377c-48d5-4cdc-8006-b9535cb0d390.png)

### Final table of errors for optimized hyperparameters
#### Good hyperparameters
![image](https://user-images.githubusercontent.com/76472249/225253292-07a24d23-3eaf-4e56-9fa7-69849ea62ce8.png)
#### Output
![image](https://user-images.githubusercontent.com/76472249/225249945-351d01c5-5392-463a-a431-eec5efe411da.png)

# Outputs for SGD with Momentum
There are 3 hyperparamters: Epochs, Ridge Lambda, Momentum. We tried to see the trend in all hyperparameters individually which was fast, and easy to analyse. Following are the outputs for each hyperparameter tuning.

### Tuning Epochs
![image](https://user-images.githubusercontent.com/76472249/225269135-40872045-146b-4dfd-a7f7-34f498bffc29.png)

### Tunning Ridge Lambda
![image](https://user-images.githubusercontent.com/76472249/225270193-82eaf256-bdf8-4de6-bbc5-5ccfdd2d7656.png)

### Tuning Momentum
![image](https://user-images.githubusercontent.com/76472249/225279942-d3c88e25-4bf7-4d17-876b-da4e5d7d628d.png)

### Final table of errors for optimized hyperparameters
![image](https://user-images.githubusercontent.com/76472249/225280342-0958de5b-ec44-418c-8f07-5a274859b94b.png)

# Outputs for the asked testcases with time taken
### fit_gradient_descent using the manually computed gradients for each of unregularized mse_loss, and mse_loss with ridge regularization
![image](https://user-images.githubusercontent.com/76472249/228420121-a5a7c692-1806-4986-b4fa-18b1f7470574.png)
<br>
![image](https://user-images.githubusercontent.com/76472249/228420325-8662c81e-0b2e-4f85-a980-92c1691891b3.png)

### fit_gradient_descent using the JAX gradients for each of unregularized mse_loss, mse_loss with LASSO regularization and mse_loss with ridge regression
![image](https://user-images.githubusercontent.com/76472249/228420530-0b76aeb5-eca3-4e67-a29f-f9193a5cc8f5.png)
<br>
![image](https://user-images.githubusercontent.com/76472249/228420664-7388e66a-0751-40e6-a6ed-8778f4b534bc.png)
<br>
![image](https://user-images.githubusercontent.com/76472249/228420817-c122949c-d961-49a7-8bde-026592ae5a32.png)

### fit_gradient_descent for running SGD on mse_loss with ridge regularization
![image](https://user-images.githubusercontent.com/76472249/228420958-36f3014f-e8cd-4036-a2a0-21d5d4a16d01.png)
<br>
![image](https://user-images.githubusercontent.com/76472249/228421831-388cc540-727c-4cc8-a7db-390d67acf1c4.png)

### fit_gradient_descent for running minibatch SGD on mse_loss with ridge regularization
![image](https://user-images.githubusercontent.com/76472249/228479326-633311ba-00b7-4208-83ff-cae461d2cf2b.png)
![image](https://user-images.githubusercontent.com/76472249/228479764-56a671a2-bea6-42cc-bccf-4a061495c7e3.png)

### fit_SGD_with_momentum for running SGD on mse_loss with ridge regularization
![image](https://user-images.githubusercontent.com/76472249/228480090-c23aa44c-4703-4390-8c69-f4486635e65d.png)
![image](https://user-images.githubusercontent.com/76472249/228480320-837c48fe-6b91-448e-ae82-e0185aaea643.png)

# Key Observations
1) Jax takes very long time to execute.
2) We cannot use l1 on normal gradient and that's why we introduced jax gradients which use some approximations in calculating gradients, and that's why
errors are a bit large in case of jax.
3) Shuffling the dataset helps to add some randomness in SGD and mini batch gradient descent.
4) Initially I added columns of 1 in the fit function, and then stored that X in self.X to use it further in the predict function. But when I was doing hyperparameter tuning I realised that the size of validation labels and train labels are different. It is because of self.X. That's why I defined a new predict function which uses X which is given in the argument. 
