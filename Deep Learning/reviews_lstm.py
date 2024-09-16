#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


#!pip install gensim


# In[2]:


from gensim.models import KeyedVectors
import json


# In[4]:


df.head()


# In[5]:


df_final = df[["reviewText", "overall"]]
df_final.head()


# In[6]:


df_final=df_final.sample(frac = 0.15)


# In[7]:


get_ipython().system('wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec')


# In[8]:


len_max = 190
df_final["user_sentiment"] = df_final["overall"].map(lambda x: "Negative" if x <= 3 else "Positive")
df_final = df_final.reset_index()
del df_final['index']
df_final.head()


# In[52]:


embdim = 300


# In[57]:


lab = np.array(df_final["user_sentiment"].map(lambda x: 1. if x == "Negative" else 0.))
sk = StratifiedKFold(n_splits=5, random_state=None)
for i, x in sk.split(lab.reshape((-1, 1)), lab):
    break

df_train, df_test = df_final.iloc[i], df_final.iloc[x]

print("Training and Testing Dataframes:", df_train.shape, df_test.shape)


# In[58]:


word = os.path.join(os.curdir, 'wiki.multi.en.vec')
print('Word to Vector...')
e_model = KeyedVectors.load_word2vec_format('wiki.multi.en.vec')


# In[59]:


word


# In[60]:


voc = list(e_model.index_to_key)
print("Size of Vocabulary in model(pretrained):", len(voc))
assert "and" in e_model
assert embdim == len(e_model["and"])
pre_weights = np.zeros((1+len(voc), embdim))

for a, b in enumerate(voc):
    pre_weights[a, :] = e_model[b]

vocab_dict = dict(zip(voc, range(1, len(voc)+1)))


# In[61]:


Matcher = re.compile(r'[a-z0-9]+') 
Words = df_final["reviewText"].map(lambda x: len(Matcher.findall(x.lower())))


# In[62]:


def RTF(reviewText):
    T = []
    words_review = Matcher.findall(reviewText.lower())
    for x, y in enumerate(words_review):
        if y not in e_model:
            continue
        if x >= len_max:
            break
        T.append(vocab_dict[y])
    if len(T) < len_max:
        padd_0 = [0.]*(len_max - len(T))
        T = padd_0 + T
    return T
def RF(row):
    A = RTF(row["reviewText"])
    B = 1. if row["user_sentiment"] == "Negative" else 0.   
    return A, B


# In[63]:


def Array(j, k):
    xid = np.arange(j.shape[0])
    np.random.shuffle(xid)
    j = j[xid, :]
    k = k[xid]
    return j, k
def generateData(data, batchSize = 128, shuffle=False):  
    while(True):
        j = []
        k = []
        for _, row in data.iterrows():
            j_, k_ = RF(row)
            j.append(j_)
            k.append(k_)   
            if len(j) > batchSize:
                temp_j, temp_k = np.array(j[:batchSize]), np.array(k[:batchSize])
                if shuffle:
                    temp_j, temp_k = Array(temp_j, temp_k)
                j, k = j[batchSize:], k[batchSize:]                    
                yield temp_j, temp_k
        if len(j) > 0:
            temp_j, temp_k = np.array(j), np.array(k)
            if shuffle:
                temp_j, temp_k = Array(temp_j, temp_k)
            yield temp_j, temp_k


# In[64]:


number_batches = 0
for i, (j, k) in enumerate(generateData(df_final, batchSize=128, shuffle=True)):
    if number_batches >= 3:
        break
    else:
        print("Batch:", i)
        assert j.shape == (128, len_max)
        assert k.shape == (128,)
        print("Expected values match")
    number_batches += 1


# In[65]:


D = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[22]:


torch.manual_seed(123)


# In[66]:


class Net(nn.Module):
    def __init__(self, embdim, h_dim, v_size, pre_weights):
        super(Net, self).__init__()
        self.embed=nn.Embedding(v_size, embdim)
        self.embed.weight.data.copy_(torch.from_numpy(pre_weights))
        self.Dropout = nn.Dropout(0.1)
        self.biLSTM1 = nn.LSTM(embdim, h_dim[0], bidirectional=True, batch_first=True)
        self.biLDropOut = nn.Dropout(0.1)
        self.dropout_1 = nn.Dropout(0.1)
        self.dense_1 = nn.Linear(2*h_dim[0], 50)
        self.relu_1 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.1)
        self.output = nn.Linear(50, 1)
        self.sig = nn.Sigmoid()
        self.h_dim = h_dim 
    def forward(self, x):
        batch_len = x.shape[0]
        out = self.embed(x)
        out = self.Dropout(out)
        out, hidden = self.biLSTM1(out)
        out = self.biLDropOut(out)
        out = self.dropout_1(out)
        out = self.dense_1(out)
        out = self.relu_1(out)
        out = self.dropout_2(out)
        out = self.output(out)
        out = self.sig(out)
        out = out.view(batch_len, -1)
        out = out[:,-1]
        return out    


# In[54]:


model = Net(embdim, [256, 128], 1+len(vocab_dict), pre_weights)
model.to(D)


# In[67]:


loss_c = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)


# In[78]:


eph=5
count= 0
cl= 5
min_loss = np.Inf
model = model.float()
model.train()
for i in range(eph):
    print("Epoch number:", i+1)
    print("Train data is being passed...")
    for j, (iputs, llels) in enumerate(generateData(df_train, batchSize=128, shuffle=True)):
        if j >= np.ceil(df_train.shape[0]/128):
            break
        count+= 1
        iputs, llels = torch.from_numpy(iputs), torch.from_numpy(llels)
        iputs, llels = iputs.to(D), llels.to(D)
        model.zero_grad()
        output = model(iputs.long())
        loss = loss_c(output.squeeze(), llels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()        
        if (j+1) % 100 == 0:
            print("Batches done:", j+1)
    print("Batches done:", j+1)
    val_losses = []
    model.eval()
    print("Test data is being passed...")
    for k, (ip, ll) in enumerate(generateData(df_test, batchSize=128, shuffle=False)):
        if k >= np.ceil(df_test.shape[0]/128):
            break
        ip, ll = torch.from_numpy(ip), torch.from_numpy(ll)
        ip, ll = ip.to(D), ll.to(D)
        out = model(ip.long())
        val_loss = loss_c(out.squeeze(), ll.float())
        val_losses.append(val_loss.item())
        if (k+1) % 100 == 0:
            print("Batches done:", k+1)
    print("Batches done:", k+1)
    model.train()
    print("Epoch number: {}/{}...".format(i+1, eph),
          "Step number: {}...".format(count),
          "Loss: {:.6f}...".format(loss.item()),
          "Validation Loss: {:.6f}".format(np.mean(val_losses)))
    if np.mean(val_losses) <= min_loss:
        torch.save(model.state_dict(), './state_dict.pt')
        print('Discrease in Validation loss ({:.6f} --> {:.6f}).  Model saved ...'.format(min_loss,np.mean(val_losses)))
        min_loss = np.mean(val_losses)


# In[79]:


model.load_state_dict(torch.load('./state_dict.pt'))
model.to(D)


# In[87]:


testing_loss = []
correct_num = 0
prob_pred = []
act = []
model.eval()
for j, (test_x, test_y) in enumerate(generateData(df_test, batchSize=128)):
    if j >= np.ceil(df_test.shape[0]/128):
        break
    test_input, test_label = torch.from_numpy(test_x), torch.from_numpy(test_y)
    test_input, test_label = test_input.to(D), test_label.to(D)
    test_output = model(test_input.long())
    test_loss = loss_c(test_output.squeeze(), test_label.float())
    testing_loss.append(test_loss.item())
    pred = torch.round(test_output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(test_label.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    correct_num += np.sum(correct)
    prob_pred.extend(test_output.cpu().squeeze().detach().numpy())
    act.extend(test_y)
    if (j+1) % 100 == 0:
        print("Batches done:", j+1)
print("Batches done:", j+1)
print("Testing loss: {:.3f}".format(np.mean(testing_loss)))
testing_acc = correct_num/len(df_test)
print("Testing accuracy: {:.3f}%".format(testing_acc*100))


# In[92]:


testing_loss = []
correct_num = 0
prob_pred = []
act = []
model.eval()
for j, (test_x, test_y) in enumerate(generateData(df_train, batchSize=128)):
    if j >= np.ceil(df_train.shape[0]/128):
        break
    test_input, test_label = torch.from_numpy(test_x), torch.from_numpy(test_y)
    test_input, test_label = test_input.to(D), test_label.to(D)
    test_output = model(test_input.long())
    test_loss = loss_c(test_output.squeeze(), test_label.float())
    testing_loss.append(test_loss.item())
    pred = torch.round(test_output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(test_label.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    correct_num += np.sum(correct)
    prob_pred.extend(test_output.cpu().squeeze().detach().numpy())
    act.extend(test_y)
    if (j+1) % 100 == 0:
        print("Batches done:", j+1)
print("Batches done:", j+1)
print("Training loss: {:.3f}".format(np.mean(testing_loss)))
testing_acc = correct_num/len(df_train)
print("Training accuracy: {:.3f}%".format(testing_acc*100))

#Below output is for training accuracy and training loss

