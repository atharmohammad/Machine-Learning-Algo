#!/usr/bin/env python
# coding: utf-8

# # Generate the Data

# In[1]:


from sklearn.datasets import make_classification 
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


X,Y = make_classification(n_classes=2 , n_samples=400,n_features=2,n_informative=2,n_redundant=0,random_state=20)


# In[3]:


Y[Y==0] = -1
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# # Defining SVM class

# In[4]:


class SVM:
    def __init__(self,C=1.0):
        self.C = C
        self.W = 0
        self.b = 0
    
    
    def hingeLoss(self,W,b,X,Y):
        m = X.shape[0]
        loss = 0.0
        
        loss += 0.5*np.dot(W,W.T)
        
        for i in range(m):
            ti = Y[i]*(np.dot(W,X[i].T) + b)
            loss += self.C*max(0,(1-ti))
        
        return loss[0][0]
    
    def fit(self,X,Y,lr=0.001,batch_size=100,maxItr=150):
        n_samples,n_features = X.shape
        n = lr
        c = self.C
        
        W  = np.zeros((1,n_features))
        b = 0
        
        #Start training 
        
        # Performing Batch Gradient Descent
        losses = []
        
        for k in range(maxItr):
            
            loss = self.hingeLoss(W,b,X,Y)
            losses.append(loss)
            # initialising the random shuffled batch
            ids = np.arange(n_samples)
            np.random.shuffle(ids)
            
            for batch_start in range(0,n_samples,batch_size):
                #gradient of W and bias

                gradw = 0
                gradb = 0

                #iterating on the random shuffled batch
                for j in range(batch_start,batch_start + batch_size):
                    if j < n_samples:
                        i = ids[j]
                        ti = Y[i]*(np.dot(W,X[i].T)  + b)

                        if ti > 1:
                            gradw += 0 
                            gradb += 0
                        else:
                            #after differentiating ti
                            gradw += c*Y[i]*X[i] 
                            gradb += c*Y[i] 

                # updating the W and b in eqn E = E - learningrate*grad
                W = W - n*W + n*gradw
                b = b + n*gradb
        
        self.W = W
        self.b = b
        
        return W[0],b,losses
            


# In[5]:


svm = SVM()


# In[6]:


W,bias,losses = svm.fit(X,Y)


# In[7]:


print( f"minimum value of loss is {losses[-1]}")


# In[8]:


plt.figure(figsize=(15,8))
plt.title("Change in Loss")
plt.plot(losses)
plt.show()


# In[9]:


def hyperplanePlot(X,Y,W,b):
    #Dummy points for plot of line
    x1 = np.linspace(-3,4,10)
    x2 = -(W[0]*x1 + b)/W[1]
    
    x_p = -(W[0]*x1 + b - 1)/W[1] # W1*X1 + W2*X2 + b = 1
    x_n = -(W[0]*x1 + b + 1)/W[1]# W1*X1 + W2*X2 + b = -1
    
    plt.figure(figsize=(15,8))
    plt.title("Hyperplane plot")
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.plot(x1,x2,label="W1*X1 + W2*X2 + b = 0")
    plt.plot(x1,x_p,label="W1*X1 + W2*X2 + b = 1")
    plt.plot(x1,x_n,label="W1*X1 + W2*X2 + b = -1")
    plt.legend()
    plt.show()


# In[10]:


hyperplanePlot(X,Y,W,bias)


# # SVM in non-Linear Dataset

# In[11]:


from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[12]:


X,Y = make_circles(n_samples=500,noise=0.02)


# In[13]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[14]:


print(X.shape,Y.shape)


# ## Defining Phi function to project the X to an outer space means transforming 2D to 3D

# In[15]:


def phi(X):
    X3 = X[:,0]**2 + X[:,1]**2
    X_ = np.zeros((X.shape[0],3))
    
    X_[:,:-1] = X
    X_[:,-1] = X3
    
    return X_


# In[16]:


X_ = phi(X)
print(X_.shape)


# In[17]:


def plot3d(X,show=True):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],zdir='z',s=20,c=Y,depthshade=True)
    if show:
        plt.show()
    return ax


# In[18]:


plot3d(X_)


# # Logistic Classifier

# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[20]:


lr = LogisticRegression()


# In[21]:


acc = cross_val_score(lr,X,Y,cv=5).mean() #accuracy for the 2D dataset


# In[22]:


print(f"Accuracy if dataset is not projected on 3d space is : {acc*100}") #Which is very low


# In[23]:


acc = cross_val_score(lr,X_,Y,cv=5).mean() #accuracy for the 3D dataset


# In[24]:


print(f"Accuracy when dataset is projected on 3d space : {acc*100}") #Far better


# # Visualising the hyperplane for the 3d projected dataset on non-linear dataset

# In[25]:


lr.fit(X_,Y)


# In[26]:


W = lr.coef_[0]
bias = lr.intercept_
print(W,bias)


# In[27]:


xx,yy = np.meshgrid(range(-2,2),range(-2,2))


# In[28]:


print(xx,yy) # initialise a matrix


# In[29]:


z = -(W[0]*xx + W[1]*yy + bias)/W[2] # findind points of z according to ax + by + cz + d = 0
print(z)


# In[30]:


ax = plot3d(X_,False)
ax.plot_surface(xx,yy,z,alpha=0.2)
plt.show()


# In[ ]:




