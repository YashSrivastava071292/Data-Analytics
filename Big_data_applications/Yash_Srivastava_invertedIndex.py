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


# In[2]:


# Reading the given textfiles. As seen from the output we have got a list which has tuples.
textFile = sc.wholeTextFiles("C:/Users/ysrivastava/BDA/Assignment/InvertedIndex/*.txt")
textFile.collect()


# In[5]:


#Creating function for manipulating the filepath from the tuple that we got in previous step. Since we require only Doc Id
#Splitting element by '/' character
def document_no(fpath):
    var = fpath.split('/')
    f_name = var[len(var)-1]
    id_doc = f_name.split('.')[0]
    return id_doc


# In[20]:


# Creating the mapper function
def map_out(mrdd):
    x = document_no(mrdd[0])
    line = mrdd[1].replace('\n',' ')
    t = line.split(' ')
    print(t)
    output = []
    for c in t:
        if c:
            output.append(((c.lower(), x.lower()), 1))
    print(output)
    return output


# In[31]:


# Creating function to format the output in the format needed.
def new_data(map):
    output = map[0] + "#"
    for x in map[1]:
        doc = "{0}:{1};".format(x[0], x[1])
        output += doc
    return output


# In[32]:


Kep_pair1 = textFile.flatMap(lambda x: map_out(x)).cache()
Kep_pair1.collect()
# From the output below, we are getting key value pairs(tuples within a tuple). This contains word and the corresponding document.


# In[33]:


# Setting up our reducer - This will give us the frequecy of each word with respect to the documents.
from operator import add
red = Kep_pair1.reduceByKey(add)
red.collect()


# In[34]:


new = red.map(lambda x: (x[0][0],(x[0][1], x[1])))
new.collect()
# Formating the output, which should keep the word as key and its corresponding information as value


# In[35]:


new2 = new.groupByKey().mapValues(lambda x: list(x))
new2.collect()
# just basic grouping using the words as key


# In[36]:


new3 = new2.sortByKey()
new3.collect()


# In[37]:


new4 = new3.map(lambda x: new_data(x)).cache()
new4.collect()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




