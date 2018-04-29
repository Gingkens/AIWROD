
# coding: utf-8

# In[2]:


import numpy as np
import scipy.io as scio




def vector_to_dictionary(params,theta,hidden_layers):
        
    end = 0
    for i in range(1,hidden_layers+2):
        start = end
        end = params['W'+str(i)].shape[0]*params['W'+str(i)].shape[1] + end
        params['W'+str(i)] = theta[start:end].reshape(params['W'+str(i)].shape)
        
        
    return params




def gradients_to_vector(gradients,hidden_layers):
    theta = None
    for i in range(1,hidden_layers+2):
        new_vector = np.reshape(gradients['dW'+str(i)], (-1,1))
        if theta is None:
            theta = new_vector
        else:
            theta = np.concatenate((theta,new_vector) , axis = 0)
    return theta





def dictionary_to_vector(params,hidden_layers):    
    theta = None
    for i in range(1,hidden_layers+2):
        new_vector = np.reshape(params['W'+str(i) ], (-1,1))
        if theta is None:
            theta = new_vector
        else:
            theta = np.concatenate((theta,new_vector) , axis = 0)
    return theta




def sigmoid_gradient(Z):
    t = sigmoid(Z)
    return t*(1-t)



def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def load_mat(filename):
    data = scio.loadmat(filename)

    return data

def get_ex4():
    data = load_mat('ex4data1.mat')
    weights = load_mat('ex4weights.mat')
    
    return data,weights
