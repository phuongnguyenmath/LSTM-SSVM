# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def phi(np.ndarray[DTYPE_t, ndim=2] Xi, np.ndarray[np.int_t, ndim=1] yi, float hp_eta, long num_states): #featuremap
    
    cdef int num_vars = Xi.shape[0]
    cdef int num_dims = Xi.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] featuremap = np.zeros(num_states*num_dims + num_states**2, dtype=DTYPE)
    cdef int i,offset,j
    
    # unaries
    for i in range(num_vars):
        idx = (yi[i] - 1)*num_dims   # change to int
        for j in range(num_dims):
            featuremap[idx+j] += Xi[i,j]
    
    #pairwise
    offset = num_states*num_dims

    #yi = yi+1
    for i in range(num_vars-1):
        idx = (yi[i + 1] - 1) + num_states*(yi[i] - 1) #change to int
        featuremap[offset+idx] += hp_eta * 1
    return featuremap



def loss(np.ndarray[np.int_t, ndim = 1] ytruth, np.ndarray[np.int_t, ndim = 1] ypredict):
    cdef int i
    cdef int res=0
    cdef int num_vars = ytruth.shape[0]
    
    for i in range(num_vars):
        res += ytruth[i] != ypredict[i]
    
    return res




@cython.profile(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
def LAD(np.ndarray[DTYPE_t, ndim = 1] w, np.ndarray[DTYPE_t, ndim = 2] Xi, np.ndarray[np.int_t, ndim=1] yi, float hp_eta, long num_states):
    
    cdef int k,t,j,maxId
    cdef np.float maxVal,sumVal, tempVar
    
    cdef int num_dims = Xi.shape[1]
    cdef np.ndarray[DTYPE_t, ndim = 2] W = w[0:num_dims*num_states].reshape((num_states, num_dims))
    cdef np.ndarray[DTYPE_t, ndim = 2] C = w[num_dims*num_states:].reshape((num_states, num_states)) 
    #cdef np.ndarray[DTYPE_t, ndim = 2] W = np.zeros((num_states, num_dims))
    #cdef np.ndarray[DTYPE_t, ndim = 2] C = np.zeros((num_states, num_states)) 
    
    
    # w_1: matrix of K vector w
    # W_2: transition matrix C
    # A - Initialise everything
    
    # A - Initialise everything
    cdef int num_samples = Xi.shape[0] # number of elements in one batch  = number of samples = K
    
    cdef np.ndarray[DTYPE_t, ndim = 2] viterbi = np.zeros((num_states,num_samples)) # initialise viterbi table
    cdef np.ndarray[DTYPE_t, ndim = 1] pre_viterbi = np.zeros((num_states)) # initialise viterbi table
    cdef np.ndarray[np.int_t, ndim = 2] best_path_table = np.zeros((num_states,num_samples)).astype(np.int) # initialise the best path table
    cdef np.ndarray[np.int_t, ndim = 1] best_path = np.zeros(num_samples).astype(np.int) # this will be your output
    cdef np.ndarray[np.int_t, ndim = 1] state = np.arange(num_states) + 1
    
    # B- appoint initial values for viterbi and best path (bp) tables
    
    viterbi[:,0] = (state != yi[0])*1 + np.dot(Xi[0,:],W.transpose())  #the first column of Viterbi matrix
      
    # C- Do the iterations for viterbi for time>0 until K
    for k in range(1, num_samples): # loop through time
        for t in range (0,num_states):  # loop through the states 
            for j in range(num_states):
                pre_viterbi[j] = viterbi[j, k-1]+ np.array(hp_eta*C[j,t])
            maxVal = pre_viterbi[0]
            maxId = 0
            for j in range(1,num_states):
                if pre_viterbi[j] > maxVal:
                    maxVal = pre_viterbi[j]
                    maxId = j
            best_path_table[t,k] = maxId
            viterbi[t,k] = maxVal
            sumVal = 0.
            for j in range(num_dims):
                sumVal += Xi[k,j]*W[t,j]
            viterbi[t,k] = viterbi[t,k] + sumVal + ((yi[k] != state[t])*1)
    # D- Back-tracking
    maxVal = 1.*viterbi[0,num_samples-1]
    maxId = 0
    for j in range(num_states):
        if viterbi[j,num_samples-1] > maxVal:
            maxVal = 1.*viterbi[j,num_samples-1]
            maxId = j
    best_path[num_samples-1] =  maxId # last state
    for k in range(num_samples-1,0,-1): # states of (last-1)th to 0th time step
        best_path[k-1] = best_path_table[best_path[k],k]
    return  best_path + 1


