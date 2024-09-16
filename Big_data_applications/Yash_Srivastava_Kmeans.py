#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
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


# Reading the text file
text = sc.textFile("C:/Users/ysrivastava/BDA/Assignment/KMeans/points.txt")
text.collect()


# In[6]:


# Passing the data points to remove the tabs
def parse(points):
    var = points.split('\t')
    result = []
    for x in var:
        value = float(x)
        result.append(value)
    return np.array(result)


# In[7]:


#Calling the parsing function
arr = text.map(lambda m: parse(m)).cache()
arr.collect()


# In[20]:


# Mapper function to find the nearest centroid
def nearest_centroid(point, centroids):
    cntrd_number = 0
    nrst_centroid = float(sys.float_info.max)
    i = 0
    while i < len(centroids):
        distance = ctrd_dist(point, centroids[i])
        if distance < nrst_centroid:
            nrst_centroid = distance
            cntrd_number = i
        i += 1    
    return cntrd_number   


# In[11]:


# Function to find distance between data points and centroid
def ctrd_dist(point, centroid):
    distance = np.sum(np.square(point - centroid)) 
    return distance


# In[10]:


def process_map(point, centroids):
    cntrd_number = nearest_centroid(point, centroids)
    return (cntrd_number, (point, 1))


# In[13]:


def cntrd_points_distance(point, centroid):
    point_distance = np.sum(np.square(centroid - point)) 
    return point_distance


# In[14]:


# Calculating distance between all points
def all_points_dist(points):
    point_distance_list = []
    for (cluster_number, point) in points:
        point_distance_list.append(cntrd_points_distance(point, cluster_number))
    return sum(point_distance_list)


# In[15]:


def mapper(map_object):
    output = (map_object[0], map_object[1][0] / map_object[1][1])
    return output


# In[16]:


def pairs(pointer1 , pointer2):
    return (pointer1[0] + pointer2[0], pointer1[1] + pointer2[1])


# In[17]:


def results(cluster_points):
    for cluster_point in cluster_points:
        print("{0}#{1}".format(round(cluster_point[0], 5), round(cluster_point[1], 5)))


# In[18]:


Dist = float(0.1)
kp = [np.array([float(2) , float(5)]), np.array([float(6), float(2)]), np.array([float(1), float(1)])]
dist2 = 1.0
type(kp)
print(kp)


# In[21]:


nearest = arr.map(lambda p: process_map(p, kp))
print(nearest.take(5))


# In[23]:


Status = nearest.reduceByKey(lambda pointer1 , pointer2: pairs(pointer1 , pointer2))
print(Status.take(4))


# In[24]:


new_points = Status.map(lambda map_object: mapper(map_object)).collect()
print(new_points)


# In[25]:


updated_distance = all_points_dist(new_points)
print(updated_distance)


# In[27]:


def change_points(new_points, kp):
    for (cluster_number, point) in new_points:
        kp[cluster_number] = point
    return kp

change_points(new_points, kp)
print(kp)


# In[30]:


while dist2 > Dist:
    print(dist2)
    near = arr.map(lambda p: process_map(p, kp))
    point_Stats = nearest.reduceByKey(lambda pointer1 , pointer2: pairs(pointer1 , pointer2))
    new_Points = point_Stats.map(lambda map_object: mapper(map_object)).collect()
    print(new_Points)
    dist2 = sum(np.sum((kp[iK] - p) ** 2) for (iK, p) in new_Points)

    for (iK, p) in new_Points:
        kp[iK] = p

results(kp)

