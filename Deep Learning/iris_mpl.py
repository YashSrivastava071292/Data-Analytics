#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip3 install torch


# In[2]:


import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


# In[3]:


#Reading the data file
iris_test = pd.read_csv("iris_test.csv", header = None)
iris_training = pd.read_csv("iris_training.csv", header = None)


# In[4]:


iris_training.shape
iris_training


# In[5]:


iris_training.columns = ['x_1', 'x_2', 'x_3', 'x_4', 'y']
iris_test.columns = ['x_1', 'x_2', 'x_3', 'x_4', 'y']


# In[6]:


df= iris_training.append(iris_test)
df.head()


# In[7]:


df.shape


# In[8]:


X = torch.tensor(df[['x_1','x_2','x_3','x_4']].values, dtype=torch.float)
y = torch.tensor(df['y'].values, dtype=torch.int)


# In[9]:


torch.manual_seed(123)
shuffle = torch.randperm(y.size(0), dtype=torch.long)

X, y = X[shuffle], y[shuffle]

split = int(shuffle.size(0)*0.7)

Xtrain, Xtest = X[shuffle[:split]], X[shuffle[split:]]
Ytrain, Ytest = y[shuffle[:split]], y[shuffle[split:]]


# In[10]:


Xtest.shape


# In[11]:


D = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[36]:


class smr(torch.nn.Module):
    def __init__(self, features, h1, h2, h3, classes):
        super(smr, self).__init__()
        self.linear1 = torch.nn.Linear(features, h1)
        self.ReLU = torch.nn.ReLU(self.linear1)
        self.linear2 = torch.nn.Linear(h1, h2)
        self.ReLU = torch.nn.ReLU(self.linear2)
        self.linear3 = torch.nn.Linear(h2, h3)
        self.ReLU = torch.nn.ReLU(self.linear3)
        self.linear4 = torch.nn.Linear(h3, classes)
        self.ReLU = torch.nn.ReLU(self.linear4)
        self.droput = torch.nn.Dropout(0.2)
        
        
    def forward(self, a):
        a = F.relu(self.linear1(a))
        x = self.droput(a)
        a = F.relu(self.linear2(a))
        x = self.droput(a)
        a = F.relu(self.linear3(a))
        x = self.droput(a)
        a = F.relu(self.linear4(a))
        x = self.droput(a)
        return a
    
model = smr(features=4,h1=5, h2=10, h3=5, classes=3).to(D)
opt = torch.optim.SGD(model.parameters(), lr=0.01)


# In[37]:


def acc(act, pred):
    accuracy = torch.sum(act.view(-1).float()==pred.float()).item()/act.size(0)
    return accuracy

Xtrain = Xtrain.to(D)
Ytrain = Ytrain.to(D)
Xtest = Xtest.to(D)
Ytest = Ytest.to(D)

total_epochs = 5000
for e in range(total_epochs):
    a = model(Xtrain)
    crit = torch.nn.CrossEntropyLoss()
    opt.zero_grad()
    loss = crit(a,Ytrain.long()) 
    loss.backward()
    opt.step()
    ta = acc(Ytrain, torch.argmax(a, dim=1))
    print('Epoch_number: %03d' % (e+1), end="")
    print(' --- Training_Accuracy: %.3f' % ta, end="")
    print('--- Loss: %.3f' % loss)


# In[35]:


a = model(Xtest)
testing_accuracy = acc(Ytest, torch.argmax(a, dim=1))
print('Test set accuracy: %.2f%%' % (testing_accuracy*100))

