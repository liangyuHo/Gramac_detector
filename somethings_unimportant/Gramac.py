#!/usr/bin/env python
# coding: utf-8

# In[10]:


import networkx as nx
import os,csv
import numpy as np

#import Build_graph as bg

# def create_graph(path):
#     return bg.Build_original_graph(path)

def get_No_vertices(G):
    return len(G.nodes())

def get_No_edges(G):
    return len(G.edges())

def get_degree(G):
    out_degree = {d[0]:d[1] for d in G.out_degree(G.nodes())}
    in_degree = {d[0]:d[1] for d in G.in_degree(G.nodes())}

    w_out_degree = { i:out_degree[i]/sum(out_degree.values()) for i in out_degree }
    w_in_degree = { i:in_degree[i]/sum(in_degree.values()) for i in in_degree }
 
    oDegree = np.mean([i for i in w_out_degree.values()])
    iDegree = np.mean([i for i in w_in_degree.values()])

    return iDegree,oDegree

def get_No_connected_components(G):
    return len(list(nx.connected_components(G.to_undirected())))

def get_No_loops(G):
    return len(list(nx.simple_cycles(G)))

def get_No_PE(G):
    PE=0
    tmp=[]
    for Edge in G.edges():
        if sorted([Edge[0],Edge[1]]) in tmp:
            PE+=1
            #print(sorted([Edge[0],Edge[1]]))
            continue
        else:
            tmp.append(sorted([Edge[0],Edge[1]]))
    return PE

def Feature_collection(path):
    Feature=[]
#     G=create_graph(path)
    G=nx.read_gpickle(path)
    Feature.append(get_No_vertices(G))
    print('# of Node Collection')
    Feature.append(get_No_edges(G))
    print('# of Edge Collection')
    Degree=get_degree(G)
    Feature.append(Degree[0])
    Feature.append(Degree[1])
    print('Degree Collection')
    Feature.append(get_No_connected_components(G))
    print('# of CC Collection')
    Feature.append(get_No_loops(G))
    print('# of Loop Collection')
    Feature.append(get_No_PE(G))
    print('# of PE Collection')

    return Feature

if __name__=='__main__':
    print(Feature_collection('../Desktop/whole_fcg/Benignware_FCG/Benignware_FCG/000d9e03fb35bc09ec77aabceff9573fe1c7982f2fcbf12d3d106b96bf84c9a7.gpickle'))










