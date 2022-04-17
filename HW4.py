import numpy as np
import time
import matplotlib.pyplot as plt
import sys

is_test = True
is_random = False
seed = 521
N = 10
dimension = 3
learning_rate = 1

def generate_data(x_mu,x_var,y_mu,y_var,n):
    data = np.zeros((n,2))
    X = np.random.normal(x_mu, x_var, n)
    Y = np.random.normal(y_mu, y_var, n)
    data[:,0] = X
    data[:,1] = Y
    return data

def logis_fun(Phi,W):
    return 1 / (1 + np.exp(-Phi@W))

def predict(X,W):
    Y = np.zeros_like(X)
    # design matrix
    Phi = np.zeros((dimension))
    for i,x in enumerate(X):
        for k in range(dimension):
            Phi[k] = x**k
        prob = logis_fun(Phi,W)
        if prob < 0.5:
            Y[i] = 0
        else:
            Y[i] = 1
    return Y

def logistic(D1,D2):
    Phi = np.zeros((dimension))
    W = np.array([[1],[1],[1]])
    A = []
    mat_len = (len(D1) + len(D2))
    # 1 / (1 + e^(-XW)) - y
    L = np.zeros((mat_len,1))
    Y = np.zeros((mat_len,1))
    for i,d in enumerate(D1):
        for k in range(dimension):
            Phi[k] = d[0]**k
        A.append(Phi.copy())
        Y[i,0] = 0
    for i,d in enumerate(D2):
        for k in range(dimension):
            Phi[k] = d[0]**k
        A.append(Phi.copy())
        Y[len(D1) + i,0] = 1
    A = np.array(A)
    for i in range(N):
        print("W : \n",W)
        for j in range(mat_len):
            L[j,0] = logis_fun(A[j],W)
        # print(L - Y)
        W = W + learning_rate* A.T@(L - Y) / mat_len
        print("Gradient : \n",learning_rate* A.T@(L - Y) / mat_len)
    
    # prediction
    predict_X = np.concatenate((D1[:,0],D2[:,0]),axis=None)
    predict_Y = predict(predict_X,W)
    print(predict_Y)

# main
if is_random:
    seed = int(time.time())
np.random.seed(seed)

D1 = generate_data(1,2,1,2,50)
D2 = generate_data(10,2,10,2,50)
logistic(D1,D2)
sys.exit()

plt.figure(figsize=(8, 6), dpi=120)
plt.title("Ground truth")
plt.scatter(D1[:,0],D1[:,1],color='r',s=15)
plt.scatter(D2[:,0],D2[:,1],color='b',s=15)
plt.show()