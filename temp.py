# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:34:12 2014

@author: kurka
"""

import networkx as nx
import json
from numpy import *
import pickle


ROOTUSER = 14594813

###################
#0 - seleciona grupo de tweets da base
from PersistencyLayer import TweetsPersister
json_fp = open('credentials.json')
cred = json.load(json_fp)

persister = TweetsPersister()
root_tweets = persister.loadTweetsOfUser(ROOTUSER) #get all root tweets

root_tweets = root_tweets[:587] #work just with 600 messages
#root_tweets = root_tweets[300:400] #work just with 600 messages

tweets_collection = []
for root_tweet in root_tweets:
    retweets = persister.loadRetweets(root_tweet['tweet_id'])
    if retweets:
        #retweets.insert(0, root_tweet) #insert root tweet at the beggining of the list
        tweets_collection.append(retweets)

users = list(set([item['user_id'] for sublist in tweets_collection for item in sublist]))
#users.remove(ROOTUSER)


#f1 = open('tweetcol.data', 'w')
#f2 = open('users.data', 'w')
#
#pickle.dump(tweets_collection, f1)
#pickle.dump(users, f2)
#
#f1.close()
#f2.close()
#    
#f1 = open('tweetcol.data', 'r')
#
#tweets_collection = pickle.load(f1)
#
#users = list(set([item['user_id'] for sublist in tweets_collection[:10] for item in sublist]))


################
# 1 - monta matriz com tweets
print "Fase 2!"

tweets_matrix = zeros((len(users), len(tweets_collection))) #empty matrix

col_num = 0
for col in tweets_collection:
    #get the positions of the users, in users list
    col_indexes = array([users.index(tweet['user_id']) for tweet in col]) 
    tweets_matrix[col_indexes, col_num] = 1
    col_num += 1
    
    
#remove root user from matrix
    

#################
#visualiza matriz como imagem
from scipy.misc import toimage

img = toimage(tweets_matrix.transpose()*255, mode='L')
img.show()




#################
#CLUSTERIZACOES

#1- kmeans sklearn

#from sklearn.cluster import KMeans
#
#k_means = KMeans(init='k-means++', n_clusters=20, n_init=10)
#k_means.fit(tweets_matrix.transpose())
#k_means_labels = k_means.labels_
#k_means_cluster_centers = k_means.cluster_centers_
#k_means_labels_unique = np.unique(k_means_labels)


####
#2- kmeans varias distancias
from kurkameans import kmeans
ncluster = 20
X = tweets_matrix.transpose()

import random    
initcentres = X[random.sample(xrange(X.shape[0]), ncluster)]
centres, xtoc, dist = kmeans(X, ncluster, initcentres=None, metric="jaccard", maxiter=100, delta=.00001, verbose=True)

   
#####
#imprime clusters
for j in range(ncluster):
    for i in where(xtoc == j)[0]:
        print tweets_collection[i][0]['text'] 
    print "\n\n"


#############


#imprime colecao toda
#for tweet in tweets_collection:
#    print ">>>>>>>>>>>>>>>>>"
#    print(tweet[0]['text'])

####################
#tag clouds

import wordcloud

TEXT = ""
for tweet in tweets_collection:
    #get the positions of the users, in users list
    TEXT += " " + tweet[0]['text']
# Separate into a list of (word, frequency).
stopwords = set([x.strip() for x in open('stopwords.txt').read().split('\n')])
words = wordcloud.process_text(TEXT, stopwords=stopwords)
# Compute the position of the words.
elements = wordcloud.fit_words(words, font_path="/usr/share/fonts/TTF/arial.ttf")
# Draw the positioned words to a PNG file.
wordcloud.draw(elements, 'test.png', font_path="/usr/share/fonts/TTF/arial.ttf", scale=2)#width=500, height=500,
        


##########
# Construcao de grafo

GClean = nx.DiGraph()
#GDirty = nx.DiGraph()

#1 - seleciona conjunto de nodes
nodes_list = set(users)


clean_edges = []
#dirty_edges = []

total = float(len(nodes_list))
count = 0
for n in nodes_list:
   dirty_edges = []
   print("%s/%s" %(count, total))
   try:   
       f = open('net/'+str(n))
   except:
       print "nao achou", n
       continue
   neighbours = json.load(f)
   for nei in neighbours:
      if nei in nodes_list: #just add if node is active
         clean_edges.append((n,nei))
      #dirty_edges.append((n, nei))
   f.close()
   #GDirty.add_edges_from(dirty_edges)
   GClean.add_edges_from(clean_edges)
   count += 1



#clean dirty graph
count = 0
total = float(len(GDirty))

iterable = GDirty.degree_iter()
for n,d in iterable:
    if n not in nodes_list and d < 2:
        GDirty.remove_node(n)
    count += 1
    print("%s/%s" %(count, total))



    
#f = open("cleanedges.data", "r")
#clean_edges = pickle.load(f)

#GClean.add_edges_from(clean_edges)
#GDirty.add_edges_from(dirty_edges)

nx.write_gexf(GClean, "folha300.gexf")
nx.write_gexf(GDirty, "folha300dirty.gexf")

##############
#deteccao de comunidades!
import sys
sys.path.insert(0, 'louvain')
import community
import networkx as nx
import matplotlib.pyplot as plt

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
G = nx.erdos_renyi_graph(30, 0.05)
G = GClean.to_undirected()
#first compute the best partition
partition = community.best_partition(G)

#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))


nx.draw_networkx_edges(G,pos, alpha=0.5)
plt.show()

#G=nx.erdos_renyi_graph(1000, 0.01)
dendo = community.generate_dendogram(G)
#for level in range(len(dendo) - 1) :
#    print "partition at level", level, "is", community.partition_at_level(dendo, level)
#    print
#    print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
#    print



#agrupar as linhas!
community_matrix = zeros((len(set(partition.values())), len(tweets_collection))) #empty matrix


#jeito 1: soma os valores de toda a comunidade
row_num = 0
for com in set(partition.values()):
    list_nodes = [node for node in partition.keys()
                                if partition[node] == com]
    row_indexes = array([users.index(user) for user in list_nodes]) 
    print len(list_nodes) #get the size of the list of nodes
    community_matrix[row_num, :] = tweets_matrix[row_indexes, :].sum(0)
    row_num += 1
    

#kmeans 2

from kmeansdist import *# kmeans

dim = 10
ncluster = 50
kmsample = 100  # 0: random centres, > 0: kmeanssample
kmdelta = .001
kmiter = 100
metric = "cosine"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
seed = 1
X = community_matrix.transpose()

randomcentres = randomsample(X, ncluster)
centres, xtoc, dist = kmeans(X, randomcentres, metric=metric, verbose=2)

for j in range(ncluster):
    for i in where(xtoc == j)[0]:
        print tweets_collection[i][0]['text'] 
    print "\n\n"
    
    
    
    
###########
#propagacao!
    
def propagate(user, graph, tweet, visited):
    #visited.append(user)
    for u, follower in graph.out_edges([user]):
        #if follower not in visited:
            #propagate(follower, graph, tweet, visited)
        tweet[users.index(follower)] += 0.2 #valor arbitrario!
        #print len(visited)
       
for tweet in tweets_matrix.T:
    for user in np.where(tweet==1)[0]:
        #get user followers and increase their values
        propagate(users[user], GClean, tweet, [])
        
        

        
        