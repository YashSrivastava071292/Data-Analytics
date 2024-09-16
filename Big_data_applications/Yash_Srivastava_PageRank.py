#!/usr/bin/env python
# coding: utf-8

# In[1]:



import findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
conf = pyspark.SparkConf().setAppName('App').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)
spark


# In[20]:


import numpy as np


# In[3]:


# Reading the text file.
text = sc.textFile("C:/Users/ysrivastava/BDA/Assignment/PageRank/input.txt", 1)
text.collect()


# In[4]:


# Creating a function to parse the line
def passed(text):
    var = text.split(' ')
    return (var[0], var[1])


# In[6]:


# Passing the text through the parser function and grouping it as per the nodes as key and values being the connections that they have
txt_lnk = text.map(lambda x: passed(x)).distinct().groupByKey().mapValues(lambda x: list(x)).cache()
txt_lnk.collect()


# In[ ]:





# In[8]:


# Counting the number of nodes.
X = txt_lnk.count()
print(X)


# In[9]:


# Generating ranks for each node
rank = txt_lnk.map(lambda node: (node[0],1.0/N))
print(rank.collect())


# In[22]:


def check_conv(old_rank, new_rank, delta= 0.1):
    conv = True
    for i in range(len(new_rank)):
        if (new_rank[i][0] == old_rank[i][0]) and ( np.abs(round(new_rank[i][1], 5) - round(old_rank[i][1], 5)) > delta):
            conv = False
    return conv


# In[28]:


old_rank = rank.collect()
while True:
    print("Old Rank - {0}".format(old_rank))
    ranks = txt_lnk.join(rank).flatMap(lambda x : [(i, float(x[1][1])/len(x[1][0])) for i in x[1][0]])
    ranks = ranks.reduceByKey(lambda x,y: x+y)
    new_rank = ranks.sortByKey().collect()
    print("New_rank - {0}".format(new_rank))
    if check_conv(old_rank, new_rank):
        print("Page ranking is completed.")
        break
    old_rank = new_rank

