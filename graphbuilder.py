import networkx as nx
import os
import json

GClean = nx.DiGraph()
GDirty = nx.DiGraph()

#1 - seleciona conjunto de nodes
nodes_list = os.listdir('net')
nodes_list = map(int, nodes_list)


clean_edges = []
dirty_edges = []

total = float(len(nodes_list))
count = 1
for n in nodes_list:
   print(str(count/total)+'%')
   f = open('net/'+str(n))
   neighbours = json.load(f)
   for nei in neighbours:
      if nei in nodes_list: #just add if node is active
         clean_edges.append((n,nei))
      dirty_edges.append((n, nei))
   f.close()
   count += 1

GClean.add_edges_from(clean_edges)
GDirty.add_edges_from(dirty_edges)

    
