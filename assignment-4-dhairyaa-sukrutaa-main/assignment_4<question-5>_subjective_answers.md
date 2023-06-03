# Plots of norm of theta v/s degree on changing N
### Conclusions: 
Following are the plots obtained : 
###### for higher values of N: 
![Plot_5__](https://user-images.githubusercontent.com/76052389/228524816-32c971a4-5a88-4477-a669-d33a1b4472e1.png)
###### for lower values of N:
![Plot_5__](https://user-images.githubusercontent.com/76052389/228527071-b73d586f-6e27-4cd7-972f-791742bd0b51.png)
###### for intermediate values of N :
![Plot_5__](https://user-images.githubusercontent.com/76052389/228525038-4af8b8be-a8eb-4312-8a1e-c46c86b60216.png)

The following conclusions can be drawn from the plots: 

1. As we increase the degree of the polynomial, the magnitude of the norm of theta generally increases as well. This is expected, as higher degree polynomials have more parameters and can fit the data more closely as seen in the previous question. 

2. For smaller data sets (e.g. N=10), the magnitude of the norm of theta vary widely depending on the specific data set used. This is because with fewer data points, there is more variability in the specific data points used to train the model. As a result, the graph shows greater increase in the norm values due to the fact that overfitting is taking place due to the introduction of noise. 

3. For larger data sets, the norm values tend to be much lower since the model is not overfitting as much. In fact, although there is an increase in the overall norm values, the increase is much lower than for lower values of N. 

