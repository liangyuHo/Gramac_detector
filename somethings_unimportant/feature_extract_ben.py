#!/usr/bin/env python
# coding: utf-8

# In[4]:


from asyncore import write
from fileinput import filename
import os,csv
from posixpath import dirname
from tqdm import tqdm
import numpy as np
import eventlet,time
import Gramac
import networkx as nx
import func_timeout
from func_timeout import func_set_timeout


def read_label():
    # read label file
    label_dict = {'BenignWare':0, 'Mirai':1, 'Tsunami':2, 'Hajime':3, 'Dofloo':4, 'Bashlite':5, 'Xorddos':6, 'Android':7, 'Pnscan':8, 'Unknown':9}
    label = {}
    threshold = {}
    with open('../dataset.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        for row in rows:
            threshold[row[0]] = row[2]
            label[row[0]] = label_dict[row[1]]
    return label

@func_set_timeout(2)
def Feature_collection(path):
    Feature=[]
    G=nx.read_gpickle(path)
    Feature.append(Gramac.get_No_vertices(G))
    Feature.append(Gramac.get_No_edges(G))
    Degree=Gramac.get_degree(G)
    Feature.append(Degree[0])
    Feature.append(Degree[1])
    Feature.append(Gramac.get_No_connected_components(G))
    Feature.append(Gramac.get_No_loops(G))
    Feature.append(Gramac.get_No_PE(G))

    return Feature

def write_csv(name,label,result):
    with open('../','a+', newline='') as file:
        writer = csv.writer(file)
        tmp=[name,label]
        tmp+=result
        writer.writerow(tmp)
def main():


    label_dict=read_label()
    GraphPath='/home/kevin/Benignware_apicall/'
    for dirPath,dirName,fileName in os.walk(GraphPath):
        fileName.sort()
        for f in tqdm(fileName):
            try:
                fpath=dirPath+f
                name=f.replace('.gpickle','')
                Result=Feature_collection(fpath)
                with open('../Ben_Result.csv','a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name,label_dict[name],Result[0],Result[1],Result[2],Result[3],Result[4],Result[5],Result[6]])
                    print(name)
                    
            except func_timeout.exceptions.FunctionTimedOut:
                f=open('Error_sample.txt','a+')
                f.write(fpath+'\n')
                f.close()
                print('fail')
                continue
            except:
                pass
                
        
            

    
if __name__=='__main__':
    main()


# In[ ]:

