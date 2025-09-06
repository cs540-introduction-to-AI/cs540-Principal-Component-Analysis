from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    #load dataset using numpy's load() function
   # Your implementation goes here!
    x=np.load(filename)#assume that we are given relative path
    x=x-np.mean(x,0)
    avergae=np.average(x)
    return x


    

def get_covariance(dataset):
    # Your implementation goes here!
    
    return np.dot(np.transpose(dataset),dataset)/(len(dataset)-1)
    

def get_eig(S, m):
    # Your implementation goes here!
    #step1: get m largest raw values from eigh() method[2 element array with 1st index being eigenvalues and 2nd index being eigenvectors]
    length=len(S)
    raw_eval,raw_evec=eigh(S,subset_by_index=[length-m,length-1])
    #step2: sort eigenvalues array in descending order and keep corresponding order for eigenvectors array as well
    index_desc=raw_eval.argsort()[::-1]#get descending ordered indices only
    eval_desc=raw_eval[index_desc]#order the eigenvalues using these indices
    evec_desc=raw_evec[:,index_desc]#order eigenvectors using these indices
    #step3: make diagonal matrix of eigenvalues
    return np.diag(eval_desc),evec_desc
   
    


def get_eig_prop(S, prop):
    # Your implementation goes here!
    raw_eval_prev,raw_evec_prev=eigh(S)
    #step1:get sum of all values to get proportion of each feature vector(image)
    sum=np.sum(raw_eval_prev)
    #step2: get those eigenvalues with proportion>prop(thus value>prop*sum)
    raw_eval,raw_evec=eigh(S,subset_by_value=[prop*sum,np.inf])
    #step3:descending order arrange eigenvalues using descending indexing
    raw_eval_indices=raw_eval.argsort()[::-1]
    #step4:get descending ordered both eigenvalues and eigenvectors
    raw_eval_return=raw_eval[raw_eval_indices]
    raw_evec_return=raw_evec[:,raw_eval_indices]
    return np.diag(raw_eval_return),raw_evec_return
    # print(len(raw_eval))
    # print(raw_eval)
    # print(sum)
    # print(len(raw_eval_prev))

def project_image(image, U):
    # Your implementation goes here!
    raise NotImplementedError

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    raise NotImplementedError
def starter():
    x=load_and_center_dataset("YaleB_32x32.npy")
    print(x.shape)
    xx=get_covariance(x)
    print(xx.shape)
    y=get_eig(xx,4)
    get_eig_prop(xx,0.07)
    
starter()
