import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import struct as st
import math

is_test = True
is_random = False
seed = 521
N = 10
dimension = 3
learning_rate = 1
thresd = 0.00001
test_num = 10

train_img_arr = []
train_label_arr = []
test_img_arr = []
test_label_arr = []
# the probability of image belongs to i
Pi = []
Mu = []
classes = []
nImg = 0
nRow = 0
nCol = 0

is_print = False
if not is_test:
    filename = {'images' : 'train-images.idx3-ubyte' ,\
                'labels' : 'train-labels.idx1-ubyte' ,\
                'test_img' : 't10k-images.idx3-ubyte' ,\
                'test_labels' : 't10k-labels.idx1-ubyte',\
                'online_test' : 'testfile.txt'}
else:
    filename = {'images' : 't10k-images.idx3-ubyte' ,\
                'labels' : 't10k-labels.idx1-ubyte',\
                'online_test' : 'testfile.txt'}

def open_img(filename):
    global nImg,nRow,nCol
    with open(filename,'rb') as imagesfile:
        imagesfile.seek(0)
        magic = st.unpack('>4B',imagesfile.read(4))
        nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images
        nRow = st.unpack('>I',imagesfile.read(4))[0] #num of rows
        nCol = st.unpack('>I',imagesfile.read(4))[0] #num of column
        
        nBytesTotal = nImg*nRow*nCol*1 #since each pixel data is 1 byte
        img_arr = np.asarray(st.unpack('>'+'B'*nBytesTotal,imagesfile.read(nBytesTotal))).reshape((nImg,nRow,nCol))
        return img_arr

def open_label(filename):
    with open(filename, 'rb') as f:
        magic, size = st.unpack('>II', f.read(8))
        label_arr = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")  
        return label_arr 

def convert_img(img_arr):
    for i,img in enumerate(img_arr):
        img[img<128] = 0
        img[img>=128] = 1
    return img_arr


def load():
    global train_img_arr,train_label_arr
    global test_img_arr,test_label_arr
    
    train_img_arr = open_img(filename['images'])
    train_label_arr = open_label(filename['labels'])
    

    if is_test:
        train_img_arr = train_img_arr[0:test_num + 100]
        train_label_arr = train_label_arr[0:test_num + 100]
        test_img_arr = train_img_arr[0:test_num]
        test_label_arr = train_label_arr[0:test_num]
    else:
        test_img_arr = open_img(filename['test_img'])
        test_label_arr = open_label(filename['test_labels'])

    # convert image to 2 bins
    train_img_arr = convert_img(train_img_arr.copy())
    test_img_arr = convert_img(test_img_arr.copy())

    print("Train img num : ",len(train_label_arr))
    print("Test img num : ",len(test_label_arr))

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

# hessian factor e^(-Xik * Wk) / (1 + e^(-Xik * Wk))^2
# x = Xik , w = Wk , both are scalar
def h_fac(x,w):
    # let E = e^(-Xik * Wk)
    E = np.exp(-x*w)
    if is_print:
        print(-x*w)
    return E/(1 + E)**2

# design matrix
def design_mat(x,y):
    phi = np.zeros(dimension)
    phi[0] = 1
    phi[1] = x
    phi[2] = y
    return phi

def predict(Data,W):
    Z = np.zeros_like(Data[:,0])
    for i,d in enumerate(Data):
        phi = design_mat(d[0],d[1])
        prob = logis_fun(phi,W)
        if prob < 0.5:
            Z[i] = 0
        else:
            Z[i] = 1
    return Z

def print_result(title,W,D1,D2):
    pred_D1 = predict(D1,W)
    pred_D2 = predict(D2,W)
    TP = len(pred_D1[pred_D1==0])
    FP = len(pred_D2[pred_D2==0])
    FN = len(pred_D1[pred_D1==1])
    TN = len(pred_D2[pred_D2==1])
    print(title,"\n\n")
    print("W:")
    for w in W:
        print(w)
    print("Confusion Matrix:")
    print("{:15}{:^20}{:^20}".format("","Predict cluster 1 ","Predict cluster 2"))
    print("{:15}{:^20}{:^20}".format("Is cluster 1",TP,FN))
    print("{:15}{:^20}{:^20}".format("Is cluster 2",FP,TN))
    print("")
    print("Sensitivity (Successfully predict cluster 1): {:.5}".format(TP / (TP + FN)))
    print("Specificity (Successfully predict cluster 2): {:.5}".format(TN / (FP + TN)))
    print("")
    print("----------------------------------------")

def logistic(D1,D2):
    # f(D) = w0 + w1*x + w2*y
    W_grad = np.array([[1],[1],[1]])
    W_newt = np.array([[1],[1],[1]])
    A = []
    mat_len = (len(D1) + len(D2))
    # 1 / (1 + e^(-XW)) - y
    L = np.zeros((mat_len,1))
    Y = np.zeros((mat_len,1))
    # D1 label 0 , D2 label 1
    for i,d in enumerate(D1):
        phi = design_mat(d[0],d[1])
        A.append(phi.copy())
        Y[i,0] = 0
    for i,d in enumerate(D2):
        phi = design_mat(d[0],d[1])
        A.append(phi.copy())
        Y[len(D1) + i,0] = 1
    A = np.array(A)

    # Gradient descent
    for i in range(N):
        pre_W = W_grad.copy()
        for j in range(mat_len):
            L[j,0] = logis_fun(A[j],W_grad)
        grad = learning_rate* A.T@(Y - L)
        W_grad = W_grad + grad
        # print("Gradient : \n",grad)
        # print("W_grad : \n",W_grad)
        if (np.all((np.abs(pre_W - W_grad) < thresd))):
            print("Gradient descent converges at iteration {}.".format(i))
            break

    # Newton's method
    DA = np.zeros_like(A)
    D = np.zeros((mat_len,mat_len))
    global is_print
    for i in range(N):
        print("Iteration : {}".format(i))
        pre_W = W_newt.copy()
        if i == 2:
            is_print = False
        else:
            is_print = False
        for j in range(mat_len):
            L[j,0] = logis_fun(A[j],W_newt)
            for k in range(dimension):
                DA[j,k] = A[j,k] * h_fac(A[j,k],W_newt[k])
            D[j,j] = np.exp(-A[j]@W_newt) / (1 + np.exp(-A[j]@W_newt))**2
        print(-A[j]@W_newt)
        grad = learning_rate* A.T@(Y - L)
        # H = AT*D*A
        # H = A.T @ np.diag(np.ravel(np.exp(-A@W_newt)/(1+np.exp(-A@W_newt))**2)) @A
        H = A.T @ D @ A
        print("Hessian matrix : \n",H)
        print("Hessian inverse : \n",np.linalg.inv(H))
        if np.linalg.det(H) != 0:
            W_newt = W_newt - learning_rate * np.linalg.inv(H)@grad
            print("descent num : \n",np.linalg.inv(H)@grad)
        else:
            W_newt = W_newt + grad
            print("descent num : \n",grad)
        
        print("W_newt : \n",W_newt)
        if (np.all((np.abs(pre_W - W_newt) < thresd))):
            print("Newton's method converges at iteration {}.".format(i))
            break

    # prediction
    predict_data = np.concatenate((D1,D2),axis=0)
    predict_Z = predict(predict_data,W_grad)
    print(predict_Z)
    print_result("Gradient descent:",W_grad,D1,D2)

def EM():
    global train_img_arr,train_label_arr
    global test_img_arr,test_label_arr
    global classes,Pi,Mu

    # initial data
    classes,counts=np.unique(train_label_arr,return_counts=True)
    Pi = np.ones_like(classes , dtype=np.float64) / len(classes)
    Mu = np.ones((len(classes),nRow*nCol), dtype=np.float64) / 2.0
    # responsibility
    gm = np.zeros((len(train_img_arr),len(classes)), dtype=np.float64)
    # E-step (calc responsibility)
    for i,img in enumerate(train_img_arr):
        label = train_label_arr[i]
        prob = np.zeros_like(classes, dtype=np.float64)
        for j,l in enumerate(classes):
            for k,pix in enumerate(img.flatten()):
                prob[j] = prob[j] + (np.log(Mu[j,k]**pix) + np.log((1-Mu[j,k]) ** (1 - pix)))
            prob[j] *= Pi[j]
        gm[i,label] = prob[label] / prob.sum()
    
    # M-step (update model parameter)
    Num = np.zeros_like(classes)
    for i,img in enumerate(train_img_arr):
        label = train_label_arr[i]
        Num[label] += 1
        Mu[label] += gm[i,label] * img.flatten()
    total_num = Num.sum()
    for i,label in enumerate(classes):
        Mu[label] /= Num[label]
        Pi[label] = Num[label] / total_num
    print(Mu[0])
    print(Pi)


# main
if is_random:
    seed = int(time.time())
np.random.seed(seed)

# Logistic Regression
D1 = generate_data(1,2,1,2,50)
D2 = generate_data(10,2,10,2,50)
# logistic(D1,D2)

# EM algorithm
load()
EM()

sys.exit()

plt.figure(figsize=(8, 6), dpi=120)
plt.title("Ground truth")
plt.scatter(D1[:,0],D1[:,1],color='r',s=15)
plt.scatter(D2[:,0],D2[:,1],color='b',s=15)
plt.show()