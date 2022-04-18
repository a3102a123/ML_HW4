from cv2 import threshold
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

is_test = True
is_random = False
seed = 521
N = 1000
dimension = 3
learning_rate = 1
thresd = 0.00001

def generate_data(x_mu,x_var,y_mu,y_var,n):
    data = np.zeros((n,2))
    X = np.random.normal(x_mu, x_var, n)
    Y = np.random.normal(y_mu, y_var, n)
    data[:,0] = X
    data[:,1] = Y
    return data

def logis_fun(phi,W):
    return 1 / (1 + np.exp(-(phi@W)))

# f(D) = w0 + x*w1 + y*w2
def f_x(x,y,W):
    return W[0] + x*W[1] + y*W[2]

# design matrix
def design_mat(x,y):
    phi = np.zeros(dimension)
    phi[0] = 1
    phi[1] = x
    phi[2] = y
    return phi


def predict(D,W):
    Z = np.zeros_like(D[:,0])
    for i,d in enumerate(D):
        phi = design_mat(d[0],d[1])
        prob = logis_fun(phi,W)
        if prob < 0.5:
            Z[i] = 1
        else:
            Z[i] = 0
    return Z

def logistic(D1,D2):
    # f(D) = w0 + w1*x + w2*y
    W = np.array([[1],[1],[1]])
    A = []
    mat_len = (len(D1) + len(D2))
    # 1 / (1 + e^(-XW)) - y
    L = np.zeros((mat_len,1))
    Y = np.zeros((mat_len,1))
    for i,d in enumerate(D1):
        phi = design_mat(d[0],d[1])
        A.append(phi.copy())
        Y[i,0] = 0
    for i,d in enumerate(D2):
        phi = design_mat(d[0],d[1])
        A.append(phi.copy())
        Y[len(D1) + i,0] = 1
    A = np.array(A)
    for i in range(N):
        pre_W = W.copy()
        for j in range(mat_len):
            L[j,0] = logis_fun(A[j],W)
        grad = learning_rate* A.T@(Y - L)
        W = W + grad
        # print("Gradient : \n",grad)
        # print("W : \n",W)
        if (np.all((np.abs(pre_W - W) < thresd))):
            print("Converge at iteration {}.".format(i))
            break
    
    # prediction
    predict_data = np.concatenate((D1,D2),axis=0)
    predict_Z = predict(predict_data,W)
    print(predict_Z)

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