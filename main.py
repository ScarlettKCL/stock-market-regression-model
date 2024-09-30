import os
import kaggle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn import tree
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
kaggle.api.authenticate()
kaggle.api.dataset_download_files("andrewmvd/sp-500-stocks", path="CIFAKE", unzip=True)
files = os.listdir("CIFAKE")
file_paths = [os.path.join("CIFAKE", file) for file in files]
stocks_df, companies_df,  index_df = pd.read_csv(file_paths[0]), pd.read_csv(file_paths[1]), pd.read_csv(file_paths[2])
index_df["Time"] = np.arange(len(index_df.index))
index_df["Lag_1"] = index_df["S&P500"].shift(1)
index_df["Lag_5"] = index_df["S&P500"].shift(5)
index_df["Lag_10"] = index_df["S&P500"].shift(10)
index_df["Future_1"] = index_df["S&P500"].shift(-1)
index_df["Date"] = pd.to_datetime(index_df["Date"])
index_df.dropna(inplace=True)
print(index_df)
features = ["Time", "Lag_1", "Lag_5", "Lag_10"]
X = index_df[features]
Y = index_df["Future_1"]
X = MinMaxScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, shuffle=False)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, shuffle=False)
""" svr_model = SVR(kernel='linear')
model = svr_model.fit(X_train, Y_train)
parametersSVR = {
        'C':[0.5, 1.0, 10.0, 50.0, 100.0, 120.0,150.0, 300.0, 500.0,700.0,800.0, 1000.0],
        'epsilon':[0, 0.1, 0.5, 0.7, 0.9],
    }
grid_search_SVR_feat = GridSearchCV(estimator=model,
                           param_grid=parametersSVR,
                           cv=TimeSeriesSplit(n_splits=10), verbose = 2
    )
model = grid_search_SVR_feat.fit(X_train, Y_train)
"""
model = SGDRegressor(alpha=0.0001, max_iter = 26000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_val)
rmse = mean_squared_error(Y_val, Y_pred, squared=False)
r2 = r2_score(Y_val, Y_pred)
print(f"Root Mean Squared Value: {rmse}, R^2 Value: {r2}")
#print(model.coef_)
plt.scatter(Y_val, Y_pred)
plt.xlabel('Actual S&P500')
plt.ylabel('Predicted S&P500')
plt.title('Actual vs Predicted S&P500 Value')
plt.show()
residuals = Y_val - Y_pred
plt.scatter(Y_pred, residuals)
plt.xlabel('Predicted S&P500')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()