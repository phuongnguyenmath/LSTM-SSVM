import numpy as np
import scipy as scp
import math
import random as rd
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from time import time


## Import compiled Struct SVM functions
from StructSVMFunctions import *




## Data Generating functions

def gen_batch(dim_batch, probab_trans, dim_input, variance = 1):
    #K = dim_batch
    num_states = probab_trans.shape[0]
    y = np.zeros(dim_batch,  dtype =  int)
    y[0] = rd.randint(1, num_states)   #generate y0
    X1 = np.random.multivariate_normal(y[0]*np.ones(dim_input -1), np.eye(dim_input-1, dtype=int)*variance)  #generate x0
    for i in range(1, dim_batch):
        yi = list(np.arange(num_states) + 1)
        y[i] = np.random.choice(yi, size = 1, p = probab_trans[y[i-1] - 1,:]).item()  #generate yi
        xi = np.random.multivariate_normal(y[i]*np.ones(dim_input -1), np.eye(dim_input-1, dtype=int)*variance)  # generate xi
        X1 = np.vstack((X1,xi))   #concatenate xi
    xc1 = np.ones((dim_batch,1), dtype = int)
    X = np.hstack((xc1,X1))
    return X,y


def gen_data(dim_batch, probab_trans, dim_input, num_batch, variance = 1):
    X = np.zeros((num_batch, dim_batch, dim_input))
    y = np.zeros((num_batch, dim_batch),  dtype =  int)
    for i in range(num_batch):
        X[i], y[i] = gen_batch(dim_batch, probab_trans, dim_input, variance)
    return X,y




## Decide function (I did not put it in Cython, although it should be easy

def decide(w, Xi, hp_eta, num_states):
    num_dims = Xi.shape[1]
    W = w[0:num_dims*num_states].reshape((num_states, num_dims))
    C = w[num_dims*num_states:].reshape((num_states, num_states)) 
    # w_1: matrix of K vector w
    # W_2: transition matrix C
    # A - Initialise everything
    
    # A - Initialise everything
    num_samples = Xi.shape[0] # number of elements in one batch  = number of samples = K
    
    viterbi = np.zeros((num_states,num_samples)) # initialise viterbi table
    best_path_table = np.zeros((num_states,num_samples)) # initialise the best path table
    best_path = np.zeros(num_samples).astype(np.int) # this will be your output
    state = np.arange(num_states) + 1
    
    # B- appoint initial values for viterbi and best path (bp) tables 
    viterbi[:,0] =  np.dot(Xi[0,:],W.transpose())  #the first column of Viterbi matrix
      
    # C- Do the iterations for viterbi for time>0 until K
    for k in range(1, num_samples): # loop through time
        for t in range (0,num_states):  # loop through the states 
            pre_viterbi = viterbi[:, k-1]+ np.array(hp_eta*C[:,t])
            best_path_table[t,k], viterbi[t,k] = max(enumerate(pre_viterbi), key=operator.itemgetter(1))
            viterbi[t,k] = viterbi[t,k] + np.dot(Xi[k,:],W[t,:].transpose()).item()
    # D- Back-tracking
    best_path[num_samples-1] =  viterbi[:,num_samples-1].argmax() # last state
    for k in range(num_samples-1,0,-1): # states of (last-1)th to 0th time step
        best_path[k-1] = best_path_table[best_path[k],k]
    return  best_path + 1

## Old version of LAD for tests.
def LADOld(w, Xi, yi, hp_eta, num_states):
    num_dims = Xi.shape[1]
    W = w[0:num_dims*num_states].reshape((num_states, num_dims))
    C = w[num_dims*num_states:].reshape((num_states, num_states)) 
    # w_1: matrix of K vector w
    # W_2: transition matrix C
    # A - Initialise everything
    
    # A - Initialise everything
    num_samples = Xi.shape[0] # number of elements in one batch  = number of samples = K
    
    viterbi = np.zeros((num_states,num_samples)) # initialise viterbi table
    best_path_table = np.zeros((num_states,num_samples)) # initialise the best path table
    best_path = np.zeros(num_samples).astype(np.int) # this will be your output
    state = np.arange(num_states) + 1
    
    # B- appoint initial values for viterbi and best path (bp) tables 

    viterbi[:,0] = (state != yi[0])*1 + np.dot(Xi[0,:],W.transpose())  #the first column of Viterbi matrix
      
    # C- Do the iterations for viterbi for time>0 until K
    for k in range(1, num_samples): # loop through time
        for t in range (0,num_states):  # loop through the states 
            pre_viterbi = viterbi[:, k-1]+ np.array(hp_eta*C[:,t])
            best_path_table[t,k], viterbi[t,k] = max(enumerate(pre_viterbi), key=operator.itemgetter(1))
            viterbi[t,k] = viterbi[t,k] + np.dot(Xi[k,:],W[t,:].transpose()).item() + ((yi[k] != state[t])*1).item()
    # D- Back-tracking
    best_path[num_samples-1] =  viterbi[:,num_samples-1].argmax() # last state
    for k in range(num_samples-1,0,-1): # states of (last-1)th to 0th time step
        best_path[k-1] = best_path_table[best_path[k],k]
    return  best_path + 1


## This was taking almost 50% of the cpu time 
def lossOld(ytruth, ypredict):
    K = ytruth.shape[0]
    return K*hamming_loss(ytruth, ypredict)

## 2 times faster
def lossOld(ytruth, ypredict):
    return 1.*sum( ytruth != ypredict)

## Old version of Phi
def phiOld(Xi, yi, hp_eta, num_states): #featuremap
    
    num_vars = Xi.shape[0]
    num_dims = Xi.shape[1]
    featuremap = np.zeros(num_states*num_dims + num_states**2)
    # unaries
    for i in range(num_vars):
        idx = int((yi[i] - 1)*num_dims)   # change to int
        featuremap[idx:(idx + num_dims)] += Xi[i]
    
    #pairwise
    offset = num_states*num_dims

    #yi = yi+1
    for i in range(num_vars-1):
        idx = int((yi[i + 1] - 1) + num_states*(yi[i] - 1))  #change to int
        featuremap[offset+idx] += 1
    featuremap[offset:] = hp_eta*featuremap[offset:]
    return featuremap


## Optimization solvers


def solverFW(X, y, hp_lambda, hp_eta, num_states, num_passes = 200, do_line_search = 1, gap_threshold = 1e-3):
    eps = 1e-12
    n = X.shape[0] #number of batches
    phi1 = phi(X[0,:,:], y[0,:], hp_eta, num_states)
    d = phi1.shape[0]  #dimension of phi
    w = np.zeros((d))
    #wMat = np.zeros((d,n))
    ell = 0
    gapList = []
    valList = []
    for k in range(num_passes):
        w_s_batch = np.zeros((n,d))
        ell_s_batch = np.zeros((n))
        for i in range(n):
            X_i = X[i]
            y_i = y[i]
            ystar_i = LAD(w, X_i, y_i, hp_eta, num_states)
            #ystar = LAD_Viterbi(w[0:3], w[3:], X_i, y_i, hp_eta)
            #print(ystar_i - ystar)
            psi_i = phi(X_i, y_i, hp_eta,num_states) - phi(X_i, ystar_i, hp_eta, num_states)
            w_s_batch[i,:] = psi_i
            loss_i = loss(y_i, ystar_i)
            ell_s_batch[i] = loss_i
        #compute value of primal objective
        pre_ws = np.mean(w_s_batch, axis = 0)
        w_s = pre_ws/hp_lambda
        ell_s = np.mean(ell_s_batch)
        val_objective =  hp_lambda/2*np.dot(w,w) + ell_s - np.dot(w,pre_ws)
        #print(val_objective)
        valList.append(val_objective)
        #compute duality gap:
        gap = hp_lambda*(np.dot(w-w_s, np.transpose(w))) - ell + ell_s
        gapList.append(gap)
        #print(gap)
        if do_line_search:
            gamma_opt = gap/(hp_lambda*np.dot(w - w_s, np.transpose(w - w_s)) + eps)
            gamma = min(max(gamma_opt,0), 1)
        else:
            gamma = 2/(k+2)
        w = (1 - gamma)*w + gamma*w_s
        ell = (1 - gamma)*ell + gamma*ell_s
    return w, gapList, valList



def solverBCFW(X, y, hp_lambda, hp_eta , num_states, num_passes = 200, do_line_search = 1, gap_threshold = 1e-3, gap_every = 10, suboptThresh = 1e-2, randPerm = True):
    eps = 1e-12
    n = X.shape[0] #number of batches
    phi1 = phi(X[0,:,:], y[0,:], hp_eta, num_states)
    d = phi1.shape[0]  #dimension of phi
    allPhi = np.zeros((n,d))
    for k in range(n):
        allPhi[k,:] = phi(X[k,:,:], y[k,:], hp_eta, num_states)
    w = np.zeros((d))
    wMat = np.zeros((n,d))
    ell = 0
    ellMat = np.zeros((n))
    gapList = []
    valList = []
    gap = suboptThresh
    val_objective = 1
    for k in range(num_passes):
        #######Break if lower than threshold
        if((gap / val_objective) < suboptThresh):
            break
        
        randIndexes = np.random.choice(n,n, replace=not randPerm)
        for dummy in range(n):
            i = randIndexes[dummy]
            X_i = X[i,:,:]
            y_i = y[i,:]
            #ystar_i = maxOracle(w, X_i, y_i, hp_eta, num_states)
            ystar_i = LAD(w, X_i, y_i, hp_eta, num_states)
            psi_i = allPhi[i,:] - phi(X_i, ystar_i, hp_eta, num_states)
            loss_i = loss(y_i, ystar_i)*1.
            #compute value of primal objective
            #pre_ws = np.mean(w_s_batch, axis = 0)
            w_s = 1/(n*hp_lambda)*psi_i
            ell_s = 1/n*loss_i
            #compute duality gap:
            gapi = hp_lambda*(np.dot(wMat[i,:]-w_s, np.transpose(w))) - ellMat[i] + ell_s
            #print(gapi)
            if do_line_search:
                gamma_opt = gapi/(hp_lambda*np.dot(wMat[i,:] - w_s, np.transpose(wMat[i,:] - w_s)) + eps)
                gamma = min(max(gamma_opt,0), 1)
            else:
                gamma = 2*n/(k+2*n)

            ########
            w = w - wMat[i,:]
            wMat[i,:] = (1-gamma )*wMat[i,:] + gamma*w_s
            w = w + wMat[i,:]
            #-------
            ell = ell - ellMat[i]
            ellMat[i] = (1-gamma )*ellMat[i] + gamma*ell_s
            ell = ell + ellMat[i]
        
        #########compute GAP and VALUE of objective function
        if((gap_every > 0) & (k%gap_every==0)):
            w_s_batch = np.zeros((n,d))
            ell_s_batch = np.zeros((n))
            for i in range(n):
                X_i = X[i,:,:]
                y_i = y[i,:]
                #ystar_i = maxOracle(w, X_i, y_i, hp_eta, num_states)
                ystar_i = LAD(w, X_i, y_i, hp_eta, num_states)
                psi_i = phi(X_i, y_i, hp_eta, num_states) - phi(X_i, ystar_i, hp_eta, num_states)
                w_s_batch[i,:] = psi_i
                loss_i = loss(y_i, ystar_i)*1.
                ell_s_batch[i] = loss_i
            #compute value of primal objective
            pre_ws = np.mean(w_s_batch, axis = 0)
            w_s = pre_ws/hp_lambda
            ell_s = np.mean(ell_s_batch)
            val_objective =  hp_lambda/2*np.dot(w,w) + ell_s - np.dot(w,pre_ws)
            #print(val_objective)
            valList.append(val_objective)
            #compute duality gap:
            gap = hp_lambda*(np.dot(w-w_s, np.transpose(w))) - ell + ell_s
            gapList.append(gap)
      
    return w, gapList, valList










