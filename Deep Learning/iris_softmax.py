#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip3 install torch


# In[61]:


import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


# In[47]:


#Reading the data file
iris_test = pd.read_csv("iris_test.csv", header = None)
iris_training = pd.read_csv("iris_training.csv", header = None)


# In[48]:


iris_training.shape
iris_training


# In[49]:


iris_training.columns = ['x_1', 'x_2', 'x_3', 'x_4', 'y']
iris_test.columns = ['x_1', 'x_2', 'x_3', 'x_4', 'y']


# In[50]:


df= iris_training.append(iris_test)
df.head()


# In[121]:


df.shape


# In[122]:


X = torch.tensor(df[['x_1','x_2','x_3','x_4']].values, dtype=torch.float)
y = torch.tensor(df['y'].values, dtype=torch.int)


# In[127]:


torch.manual_seed(123)
shuffle = torch.randperm(y.size(0), dtype=torch.long)

X, y = X[shuffle], y[shuffle]

split = int(shuffle.size(0)*0.7)

Xtrain, Xtest = X[shuffle[:split]], X[shuffle[split:]]
Ytrain, Ytest = y[shuffle[:split]], y[shuffle[split:]]


# In[130]:


Xtest.shape


# In[131]:


D = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[229]:


class smr(torch.nn.Module):
    def __init__(self, features,classes):
        super(smr, self).__init__()
        self.linear1 = torch.nn.Linear(features, classes)
        self.droput = torch.nn.Dropout(0.2)
        
        
    def forward(self, a):
        log = self.linear1(a)
        prob = F.softmax(log, dim=1)
        return log, prob
        x = self.droput(a)
model = smr(features=4, classes=3).to(D)
opt = torch.optim.Adam(model.parameters(), lr=0.01)


# In[230]:


def acc(act, pred):
    accuracy = torch.sum(act.view(-1).float()==pred.float()).item()/act.size(0)
    return accuracy

Xtrain = Xtrain.to(D)
Ytrain = Ytrain.to(D)
Xtest = Xtest.to(D)
Ytest = Ytest.to(D)

total_epochs = 500
for e in range(total_epochs):
    log,prob = model(Xtrain)
    loss = F.cross_entropy(log, Ytrain.long())
    opt.zero_grad()
    loss.backward()
    opt.step()
    log, prob = model(Xtrain)
    a = acc(Ytrain, torch.argmax(prob, dim=1))
    print('Epoch_number: %03d' % (e+1), end="")
    print(' --- Training_Accuracy: %.3f' % a, end="")
    print('--- Loss: %.3f' % F.cross_entropy(log, Ytrain.long()))


# In[ ]:





# In[193]:


log, prob = model(Xtest)
testing_accuracy = acc(Ytest, torch.argmax(prob, dim=1))
print('Test set accuracy: %.2f%%' % (testing_accuracy*100))

