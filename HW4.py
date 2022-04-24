import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import struct as st
import math
import time

is_EM = True
is_test = False
is_random = False
seed = 521
N = 5000
EM_N = 3
learning_rate = 0.01
print_num = 4
thresd = 0.001
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
dimension = 3

is_print = False
if not is_test:
    filename = {'images' : 'train-images.idx3-ubyte' ,\
                'labels' : 'train-labels.idx1-ubyte' ,\
                'test_img' : 't10k-images.idx3-ubyte' ,\
                'test_labels' : 't10k-labels.idx1-ubyte'}
else:
    filename = {'images' : 't10k-images.idx3-ubyte' ,\
                'labels' : 't10k-labels.idx1-ubyte'}

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
        test_img_arr = train_img_arr[test_num + 100:test_num * 2 + 100]
        test_label_arr = train_label_arr[test_num + 100:test_num * 2 + 100]
        train_img_arr = train_img_arr[0:test_num + 100]
        train_label_arr = train_label_arr[0:test_num + 100]
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
    return 1 / (1 + np.exp(np.clip(-(phi@W),-500,500)))

# f(D) = w0 + x*w1 + y*w2
def f_x(x,y,W):
    return W[0] + x*W[1] + y*W[2]

# hessian factor e^(-Xik * Wk) / (1 + e^(-Xik * Wk))^2
# x = Xik , w = Wk , both are scalar
def h_fac(x,w):
    # let E = e^(-Xik * Wk)
    E = np.exp(np.clip(-x*w,-500,500))
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

def draw_result(title,D1,D2,ax):
    ax.set_title(title)
    ax.scatter(D1[:,0],D1[:,1],color='r',s=15)
    ax.scatter(D2[:,0],D2[:,1],color='b',s=15)

def logistic(D1,D2):
    # f(D) = w0 + w1*x + w2*y
    W_grad = np.array([[1],[1],[1]])
    W_newt = np.array([[1],[1],[1]])
    descent_num = np.array([[0],[0],[0]])
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
    normalize_A = NormalizeData(A)

    # Gradient descent
    for i in range(N):
        pre_W = W_grad.copy()
        for j in range(mat_len):
            L[j,0] = logis_fun(A[j],W_grad)
        pre_desc_num = descent_num.copy()
        grad = learning_rate * A.T@(Y - L)
        descent_num = grad
        W_grad = W_grad + descent_num
        if i <= print_num or i >= N-5:
            print("Gradient : \n",descent_num)
            print("W_grad : \n",W_grad)
        if (np.sum((np.abs(pre_W - W_grad) < thresd)) >=1):
            print("Gradient descent converges at iteration {}.".format(i))
            break
        elif i == N-1:
                print("Newton's method converges at iteration {}.".format(i))

    print("Begin newton's method\n\n")
    # Newton's method
    D = np.zeros((mat_len,mat_len), dtype=np.float64)
    global is_print
    for i in range(N):
        # print("Iteration : {}".format(i))
        pre_W = W_newt.copy()
        if i == 2:
            is_print = False
        else:
            is_print = False
        for j in range(mat_len):
            L[j,0] = logis_fun(A[j],W_newt)
            # normlize_W = W_newt / np.linalg.norm(W_newt)
            # normalize_A = A[j] / np.linalg.norm(A[j])
            # D[j,j] = np.exp(-A[j]@W_newt) / (1 + np.exp(-A[j]@W_newt))**2
            if j == mat_len - 1 and False:
                print(np.exp(-normalize_A[0]@W_newt)/(1 + np.exp(-normalize_A[0]@W_newt)),1/(1 + np.exp(-normalize_A[0]@W_newt)))
            logis = logis_fun(A[j],W_newt)
            D[j,j] = logis * (1 - logis) - 0.5
        grad = A.T@(Y - L)
        # H = AT*D*A
        # H = A.T @ np.diag(np.ravel(np.exp(-A@W_newt)/(1+np.exp(-A@W_newt))**2)) @A
        H = A.T @ D @ A
        if np.linalg.det(H) != 0:
            H_inv = np.linalg.inv(H)
            pre_desc_num = descent_num.copy()
            descent_num = 1 * (H_inv@grad)
            W_newt = W_newt - descent_num
        else:
            W_newt = W_newt + grad
        # print message
        if i <= print_num or i >= N-5:
            print("Hessian matrix : \n",H)
            print("Hessian inverse : \n",np.linalg.inv(H))
            print("Gradient : \n",grad)
            if np.linalg.det(H) != 0:
                print("descent num : \n",-np.linalg.inv(H)@grad)
            else:
                print("descent num (by gradient descent): \n",grad)
            print("W_newt : \n",W_newt)
        if (np.sum((np.abs(pre_W - W_newt) < thresd)) >= 2):
        # if (np.all((np.abs(pre_desc_num - descent_num) < thresd))):
            print("Newton's method converges at iteration {}.".format(i))
            break
        elif i == N-1:
            print("Newton's method converges at iteration {}.".format(i))

    # prediction
    predict_data = np.concatenate((D1,D2),axis=0)
    predict_grad_Z = predict(predict_data,W_grad)
    predict_new_Z = predict(predict_data,W_newt)
    predict_grad_Z_idx = (predict_grad_Z == 1)
    predict_new_Z_idx = (predict_new_Z == 1)
    print("Gradient descent prediction :",predict_grad_Z)
    print("Newton's method prediction : \n",predict_new_Z)
    print_result("Gradient descent:",W_grad,D1,D2)
    print_result("Newton's method descent:",W_newt,D1,D2)
    
    # draw result
    # plt.figure(figsize=(8, 6), dpi=120)
    fig, axs = plt.subplots(1,3)
    fig.set_figheight = 8
    fig.set_figwidth = 6
    fig.dpi = 150
    grad_D1 = predict_data[np.invert(predict_grad_Z_idx)]
    grad_D2 = predict_data[predict_grad_Z_idx]
    new_D1 = predict_data[np.invert(predict_new_Z_idx)]
    new_D2 = predict_data[predict_new_Z_idx]
    draw_result("Ground truth",D1,D2,axs[0])
    draw_result("Gradient descent",grad_D1,grad_D2,axs[1])
    draw_result("Newton's method",new_D1,new_D2,axs[2])

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def log_Bernoulli(p,x):
    p1 = p**x
    p0 = (1 - p) ** (1-x)
    return np.log(p1) + np.log(p0)
    (np.log(Mu[j,k]**pix) + np.log((1-Mu[j,k]) ** (1 - pix)))

# D1 : prediction , D2 : ground turth
def print_EM_result(D1,D2,label=""):

    D1 = (D1 == 1)
    D2 = (D2 == 1)
    inv_D1 = np.invert(D1)
    inv_D2 = np.invert(D2)
    TP = np.logical_and(D1,D2).sum()
    FP = np.logical_and(D1,inv_D2).sum()
    FN = np.logical_and(inv_D1,D2).sum()
    TN = np.logical_and(inv_D1,inv_D2).sum()

    print("Confusion Matrix {}:".format(label))
    print("{:15}{:^20}{:^20}".format("","Predict number {} ".format(label),"Predict not number {}".format(label)))
    print("{:15}{:^20}{:^20}".format("Is number {}".format(label),TP,FN))
    print("{:15}{:^20}{:^20}".format("Is number {}".format(label),FP,TN))
    print("")
    print("Sensitivity (Successfully predict number {}): {:.5}".format(label,TP / (TP + FN)))
    print("Specificity (Successfully predict not number {}): {:.5}".format(label,TN / (FP + TN)))
    print("")
    print("----------------------------------------")

def EM():
    global train_img_arr,train_label_arr
    global test_img_arr,test_label_arr
    global classes,Pi,Mu

    # initial data
    classes,counts=np.unique(train_label_arr,return_counts=True)
    # the probability of label i appear
    Pi = np.ones_like(classes , dtype=np.float64) / len(classes)
    # the probablity of pixel shows 1
    Mu = np.ones((len(classes),nRow*nCol), dtype=np.float64) / 2.0
    # responsibility
    gm = np.zeros((len(train_img_arr),len(classes)), dtype=np.float64)
    for ite in range(EM_N):
        strat_time = time.time()
        print("{} iteration : ".format(ite))
        # E-step (calc responsibility)
        # prevent zero show in log function
        Mu[Mu == 0] = 1e-50
        Mu[Mu >= 1.0] = 1 - 1e-10
        for i,img in enumerate(train_img_arr):
            label = train_label_arr[i]
            prob = np.zeros_like(classes, dtype=np.float64)
            for j,l in enumerate(classes):
                for k,pix in enumerate(img.flatten()):
                    prob[j] = prob[j] + (np.log(Mu[j,k]**pix) + np.log((1-Mu[j,k]) ** (1 - pix)))
                prob[j] += np.log(Pi[j])
            # print(prob,prob.sum())
            prob = np.exp(prob)
            gm[i,label] = prob[label] / prob.sum()
        # print(gm[0])
        
        # M-step (update model parameter)
        Num = np.zeros_like(classes)
        Mu.fill(0)
        for i,img in enumerate(train_img_arr):
            label = train_label_arr[i]
            Num[label] += 1
            Mu[label] += gm[i,label] * img.flatten()
        total_num = Num.sum()
        for i,label in enumerate(classes):
            Mu[label] /= Num[label]
            Pi[label] = Num[label] / total_num
        # plt.imshow(Mu[5].reshape(img.shape),cmap='gray')
        # plt.show()
        # for l in classes:
        #     print(l," max : ",Mu[l].max()," / min : ",Mu[l].min())
        # print(Pi)
        end_time = time.time()
        time_c= end_time - strat_time
        min_c = int(time_c / 60)
        print('Iteration {} time cost : {}m , {:.3f}s'.format(ite,min_c,time_c))

def predict_EM():
    global test_img_arr,test_label_arr
    global classes,Pi,Mu
    prob = np.zeros_like(classes, dtype=np.float64)
    result = np.zeros((len(test_img_arr)))
    # prevent zero show in log function
    Mu[Mu == 0] = 1e-50
    Mu[Mu >= 1.0] = 1 - 1e-10
    print("Pi : ",Pi)
    for i,img in enumerate(test_img_arr):
        prob.fill(0)
        for j,l in enumerate(classes):
            for k,pix in enumerate(img.flatten()):
                prob[j] = prob[j] + (np.log(Mu[j,k]**pix) + np.log((1-Mu[j,k]) ** (1 - pix)))
                if i == 0 and j == 0 and k > 500:
                    print("mu : {} , pix : {}".format(Mu[j,k],pix))
                    print(prob[j],Mu[j,k]**pix,(1-Mu[j,k]) ** (1 - pix))
            prob[j] += np.log(Pi[j])
        result[i] = np.argmax(prob)
        print(prob)

    for l in classes:
        groundTruth = test_label_arr.copy()
        prediction = result.copy()
        trueTable = (groundTruth == l)
        groundTruth[trueTable] = 1
        groundTruth[np.invert(trueTable)] = 0
        trueTable = (prediction == l)
        prediction[trueTable] = 1
        prediction[np.invert(trueTable)] = 0
        print_EM_result(prediction,groundTruth,l)

    print(test_label_arr)
    print(result)


# show the image
def im_show(img,label,ax):
    ax.set_title('Label is {label}'.format(label=label))
    ax.imshow(img, cmap='gray')

def draw_EM():
    global classes,Pi,Mu
    fig, axs = plt.subplots(2, 5)
    for a in range(0,2):
        for b in range(0,5):
            ax = axs[a,b]
            l = classes[a*5+b]
            img = np.zeros_like(test_img_arr[0])
            row,col = img.shape
            for i in range(row):
                for j in range(col):
                    k = i*col + j
                    prob0 = 1-Mu[l,k]
                    prob1 = Mu[l,k]
                    if prob1 >= prob0:
                        img[i,j] = 1
            print("Class {}:".format(l))
            print(img)
            im_show(img,l,ax)

# main
if is_random:
    seed = int(time.time())
np.random.seed(seed)

strat_time = time.time()
if not is_EM:
    # Logistic Regression
    if is_test:
        D1 = generate_data(1,2,1,2,50)
        D2 = generate_data(10,2,10,2,50)
        D1 = generate_data(1,2,1,2,50)
        D2 = generate_data(3,4,3,4,50)
    else:
        print("Point number : ")
        point_num = int(input())
        print("input mx1 : ")
        mx1 = float(input())
        print("input my1 : ")
        my1 = float(input())
        print("input mx2 : ")
        mx2 = float(input())
        print("input my2 : ")
        my2 = float(input())
        print("input vx1 : ")
        vx1 = float(input())
        print("input vy1 : ")
        vy1 = float(input())
        print("input vx2 : ")
        vx2 = float(input())
        print("input vy2 : ")
        vy2 = float(input())
        D1 = generate_data(mx1,vx1,my1,vy1,point_num)
        D2 = generate_data(mx2,vx2,my2,vy2,point_num)
    logistic(D1,D2)
else:
    # EM algorithm
    load()
    EM()
    predict_EM()
    draw_EM()
end_time = time.time()
time_c= end_time - strat_time
min_c = int(time_c / 60)
print('Total time cost : {}m , {:.3f}s'.format(min_c,time_c))
plt.show()