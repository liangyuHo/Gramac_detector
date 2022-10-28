import numpy as np
import r2pipe, os, sys
import json
from pwn import *
import pandas as pd
from param_parser import parameter_parser
import networkx as nx
import pickle
import copy


def create_graph(path):
    r2 = r2pipe.open(path)
    r2.cmd('aaaa')
    funcCall=r2.cmdj('agCj')
    print(funcCall)
    G = nx.DiGraph()
    for i in range(len(funcCall)):
        cur_node = funcCall[i]['name']
        G.add_node(cur_node)
        for j in range(len(funcCall[i]['imports'])):
            G.add_edge(cur_node,funcCall[i]['imports'][j])
            
    try:
        temp = copy.deepcopy(G.nodes())
        for d in temp:
            if d[:3] == 'fcn':
                G.remove_node(d)
    except:
        print('fail to convert the CFG to api call graph')        
        return 0

    return G

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
    
    try:
        G=create_graph(path)
    except:
        print('fail to constuct the FCG.')
        return 0
    Feature.append(get_No_vertices(G))
    Feature.append(get_No_edges(G))
    Degree=get_degree(G)
    Feature.append(Degree[0])
    Feature.append(Degree[1])
    Feature.append(get_No_connected_components(G))
    Feature.append(get_No_loops(G))
    Feature.append(get_No_PE(G))
    return np.array(Feature)



def main(args):
        
    try:
        feature = Feature_collection(args.input_path)
    except:
        print('fail to collect the feature.')
        return 0

    np.save('./feature/feature',feature)
    

if __name__=='__main__':
    args = parameter_parser()
    main(args)

