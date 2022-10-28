import numpy as np
import r2pipe, os, sys
import json
from pwn import *
import networkx as nx
from interruptingcow import timeout
import pandas as pd
import csv

def FCG_extraction(path,kind,name):
  start= time.time()
  r2 = r2pipe.open(path)
  r2.cmd('aaaa')
  # Extraction
  funcCall=r2.cmdj('agCj')
  G = nx.DiGraph()
  for i in range(len(funcCall)):
    cur_node = funcCall[i]['name']
    G.add_node(cur_node)
    for j in range(len(funcCall[i]['imports'])):
      G.add_edge(cur_node,funcCall[i]['imports'][j])
  print(G.number_of_nodes(), G.number_of_edges())
  if kind=="BenignWare":
    os.chdir("/home/b10704118/OpcodeTask/Benignware_FCG")
    nx.write_gpickle(G,name+".gpickle" )
    os.chdir("/home/b10704118/OpcodeTask")
  else:
    os.chdir("/home/b10704118/OpcodeTask/Malware_FCG")
    nx.write_gpickle(G,name+".gpickle" )
    os.chdir("/home/b10704118/OpcodeTask")
  end= time.time()
  exe_time = end - start
  if G.number_of_nodes()>0:
    state="Success"
    if kind=="BenignWare":
      with open('BenignWare_FCG.csv', 'a', newline='') as data:
        writer = csv.writer(data)
        writer.writerow([name,state,exe_time,G.number_of_nodes(),G.number_of_edges()])
    else:
      with open('Malware_FCG.csv', 'a', newline='') as data:
        writer = csv.writer(data)
        writer.writerow([name,state,exe_time,G.number_of_nodes(),G.number_of_edges()]) 
  else:
    state="Fail"
    if kind=="BenignWare":
      with open('BenignWare_FCG.csv', 'a', newline='') as data:
        writer = csv.writer(data)
        writer.writerow([name,state,exe_time,-1,-1])
    else:
      with open('Malware_FCG.csv', 'a', newline='') as data:
        writer = csv.writer(data)
        writer.writerow([name,state,exe_time,-1,-1])
  
def main():
  data = pd.read_csv('Malware_Opcode_Sequence_fixed_size.csv',header=None)
  kind = data.iloc[:,1]
  kind = kind.to_numpy()
  filename = data.iloc[:,0]
  filename = filename.to_numpy()
  #批次
  for i in range(140000,150000):
    if kind[i]=='BenignWare':
      print("start:",i+1,"   Extract benignWare")
      input_path = "../../../../dataset/benignware/"+filename[i][0]+filename[i][1]+"/"+filename[i]
    else:
      print("start:",i+1,"   Extract malware")
      input_path = "../../../../dataset/linuxmal/"+filename[i][0]+filename[i][1]+"/"+filename[i]
    #print(input_path)
    FCG_extraction(input_path,kind[i],filename[i])

if __name__=='__main__':
    main()
