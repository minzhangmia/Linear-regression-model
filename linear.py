import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalization(x):
    mean = x.mean()
    std = x.std()
    x = (x - mean) / std
    return x, mean, std


def gradientDescentLinearRegression(X, y, iterations = 1000, learning_rate = 0.01):
    J_history = []
    b = 0
    m = 0
    n = X.shape[0]
    for _ in range(iterations):
        b_gradient = 2 * np.sum((m*X + b) - y) / float(n)
        m_gradient = 2 * np.sum(X*((m*X + b) - y)) / float(n)        
        b = b - (learning_rate * b_gradient)
        m = m - (learning_rate * m_gradient)
        cost = np.sum((m*X+b-y)*(m*X+b-y)) / float(n)
        J_history.append(cost)
    return m, b, J_history


def plotCost(J_history, iterations):
    x = np.arange(1,iterations+1)
    plt.plot(x,J_history)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Cost of iterations")
    plt.show()


df = pd.read_csv("week1.csv")
X=np.array(df.iloc[: , 0])
y=np.array(df.iloc[: , 1])
norm_X, xmean, xstd = normalization(X)
norm_y, ymean, ystd = normalization(y)
learning_rate = 0.01
iterations = 1000

m,b,J_history = gradientDescentLinearRegression(norm_X, norm_y, iterations, learning_rate)
print(m,b)


plt.scatter(X, y)
plt.plot(X, ystd*(m*norm_X + b)+ymean)
print(ystd)
print(ymean)

plt.title("Gradient Descent Linear Regressor")
plt.show()

plotCost(J_history, iterations)
