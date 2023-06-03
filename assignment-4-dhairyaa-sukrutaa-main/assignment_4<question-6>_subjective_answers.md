# Plots of errors with normalized as well as unnormalized data
We have plotted the errors over N values for different input data and following are results: 
In each of the cases y in dependent on the first and the second arguements of x linearly. 
### 1 : x is randomly generated over a normal distribution for varying n values and has 10 columns, all of the columns have mean set to 0 and variance set to 1. 
![Plot_6_0](https://user-images.githubusercontent.com/76472249/228488792-f28aae81-2da9-4b8d-8a28-57633bdf5da2.png)
In such as case since the input data is already normalised, the errors obtained for normalised vs unnormalised input data is almost the same and it can be seen from the above graph. 
### 2 : x is randomly generated over a normal distribution for varying n values and has 10 columns, only the columns that x is depend on have high variance, while the others have lower variance values. All the columns in X have lower mean values. 
It can be noticed that the errors obtained through normalisation is lesser than those obtained from unnormalised data. This is due to the fact that if the model is predicted over the unnormalised dataset, the high variance points might we given lower importance and the function could return wrong values. However since normalising the output can increase the importance of high variance data points, it gives better outputs. 
![Plot_6_1](https://user-images.githubusercontent.com/76472249/228489005-a6c7ed17-0eb0-4f03-974e-a3071e422d2a.png)
### 3 : x is randomly generated over a normal distribution for varying n values and has 10 columns, only the columns that x is depend on have high variance, while the others have lower variance values. All the columns in X have higher and varying mean values. 
It can be noticed that normalisation can give vastly better values in this case as compared to unnormalised inputs.This is due to the fact that the model will fit the data badly due to the varied means and higher variance for the columns that y is actually dependent on. Normalization will not only increase the imporatnce of high variance terms, it will also make the trends in the input data vs the output data much more clear using a much better fit. 
![Plot_6_highmean_1](https://user-images.githubusercontent.com/76472249/228489071-c7e4ee57-38b0-4342-bb46-b537193ba66a.png)
