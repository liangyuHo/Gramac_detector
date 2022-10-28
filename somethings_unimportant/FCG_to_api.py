#!/usr/bin/env python
# coding: utf-8

# In[3]:


import networkx as nx
import copy
import os

dirpath = "../Desktop/whole_fcg/Malware_FCG/Malware_FCG/"
result = os.listdir(dirpath)
# Gtest = nx.read_gpickle(dirpath+result[0])

for index in range(len(result)):
    print(index)
    try:
        Gtest = nx.read_gpickle(dirpath+result[index])
        temp = copy.deepcopy(Gtest.nodes())
        for d in temp:
            if d[:3] == 'fcn':
                Gtest.remove_node(d)
#         print(Gtest.nodes())
    except:
        print('error-------------------------')


# In[10]:


print(Gtest.nodes())


# In[ ]:




