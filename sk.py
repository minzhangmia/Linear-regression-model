import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("week1.csv")
X=np.array(df.iloc[: , 0])
X=X.reshape(-1,1)
y=np.array(df.iloc[: , 1])
y=y.reshape(-1,1)

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()   
scaler.fit(X) #Compute the mean and std to be used for later scaling.
yscaler = StandardScaler()   
yscaler.fit(y)
x_train = scaler.transform(X)
y_train = yscaler.transform(y)
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
print(model.coef_)
print(model.intercept_)
print(yscaler.var_)
print(yscaler.mean_)

plt.scatter(X, y)
plt.plot(X, math.sqrt(yscaler.var_)*((model.coef_*x_train) + model.intercept_)+yscaler.mean_)
plt.title("Gradient Descent Linear Regressor")
plt.show()

print(X)
print(y)
