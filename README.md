# stock-market-regression-model
## Project to investigate the effectiveness of different regression model algorithms of predicting the future value of the S&amp;P500.
Can be run in a notebook or any environment that supports Python.
### About
This project was carried out to develop my understanding of how to implement different linear regression algorithms in supervised machine learning applications, as well as to investigate which model would be the most effective at predicting future values of the S&P500, with the dataset used origination from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks/data?select=sp500_companies.csv), by comparing the root mean squared and R^2 values that each model produced when applied to the validation data.

### Model Statistics
DecisionTreeRegressor(random_state=0): 
Root Mean Squared Value: 689.2075219059914, R^2 Value: -0.5377247291191039
![S P500 AVP DT](https://github.com/user-attachments/assets/76dbfc7f-372f-4a4f-b95d-01f1ea612a19)
![S P500 RP DT](https://github.com/user-attachments/assets/b07f150a-ce9f-49da-ba2b-a172fe4cc71c)

LinearRegression():
Root Mean Squared Value: 53.60479816291607, R^2 Value: 0.9906977948846726
![S P500 AVP SVR](https://github.com/user-attachments/assets/b6acadfe-609d-4827-b2a2-5385d77cf7c2)


LassoCV(n_alphas=5000, max_iter=5000, random_state=0):
Root Mean Squared Value: model = 53.872969368211585, R^2 Value: 0.9906044889296942
![S P500 AVP L](https://github.com/user-attachments/assets/9674b96f-e63e-4e97-a535-fc87a409a6dc)
![S P500 RP L](https://github.com/user-attachments/assets/ffe99abf-2af2-4142-8649-b7e1e1a2dd1d)

RidgeCV():
Root Mean Squared Value: 68.59297020691136, R^2 Value: 0.9847686709877976
![S P500 AVP R](https://github.com/user-attachments/assets/bbdadeb4-6c13-4583-abe8-a9992a8677c7)
![S P500 RP R](https://github.com/user-attachments/assets/8f8458b8-55a9-4c08-8f68-6b6d71714500)

SGDRegressor(alpha=0.0001, max_iter = 26000):
Root Mean Squared Value: 77.66649684914823, R^2 Value: 0.9804725284034541
![S P500 AVP SGD](https://github.com/user-attachments/assets/0b1d19e2-2072-450a-9329-1a2a2ca09f0b)
![S P500 RP SGD](https://github.com/user-attachments/assets/fe5eb5fd-9708-4d16-b1b8-d83b4ac7e346)

### Evaluation
Using the values of the metrics calculated for each model, it was found that the Support Vector Regression (SVR/LinearRgression()) model performed the best, given that it had the lowest Root Mean Squared value of 53.60, and the highest R^2 value of 0.991. Apart from the DecisionTreeRegressor model, which was shown to be unsuitable for the data given that it had an extremely high root Mean Squared value and low R^2 value, the other models performed reasonably well, having R^2 values fairly similar to the SVR model, with the LassoCV model performing second best given that its Root Mean Squared value was only marginally behind the SVR model's. To improve this model's performance, the parameters of the model could be experimented with and adjusted to the optimal values, or more features could be engineered from the data to be used. 
