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
#root_tweets = persister.loadTweetsOfUser(ROOTUSER) #get all root tweets


f = open("folha2600classes.data", "r")
root_tweets_extended = pickle.load(f)
f.close()

def topiccleaner(tweet):
    if tweet['class'] in ['cotidiano', 'esporte', 'mundo', 'poder', 'ilustrada', 'mercado']:
        return True
    else:
        return False

#cleaning root tweets
root_tweets = filter(topiccleaner, root_tweets_extended)




tweets_collection = []
classes = []
for root_tweet in root_tweets:
    retweets = persister.loadRetweets(root_tweet['tweet']['tweet_id'])
    if retweets and len(retweets)>=10: #limiting just stories with more than 10 retweets
        #retweets.insert(0, root_tweet) #insert root tweet at the beggining of the list
        tweets_collection.append(retweets)
        classes.append(root_tweet['class'])


classes = np.array(classes)
tweets_collection = np.array(tweets_collection)
users = np.array(list(set([item['user_id'] for sublist in tweets_collection for item in sublist])))
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
    col_indexes = array([np.where(users == tweet['user_id']) for tweet in col]) 
    tweets_matrix[col_indexes, col_num] = 1
    col_num += 1
    
#remove bots from matrix
rsum = np.sum(tweets_matrix, axis=1)
bots = np.argsort(rsum)[-1]

tweets_matrix = np.delete(tweets_matrix, bots, axis=0)
users = np.delete(users, bots)



#clear empty lines or colums
rsum = np.sum(tweets_matrix, axis=1)
zr = np.where(rsum==0)
tweets_matrix = np.delete(tweets_matrix, zr, axis=0)
users = np.delete(users, zr)
    
lsum = np.sum(tweets_matrix, axis=0)
zl = np.where(lsum<10)
tweets_matrix = np.delete(tweets_matrix, zl, axis=1)
tweets_collection = np.delete(tweets_collection, zl)
classes = np.delete(classes, zl)


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
centres, xtoc, dist = kmeans(X, ncluster, initcentres=None, metric="cosine", maxiter=100, delta=.00001, verbose=True)

   
#####
#imprime clusters
cols = []
for j in range(ncluster):
    for i in where(xtoc == j)[0]:
        print tweets_collection[i][0]['text'][14:] 
        cols.append(i)
    print "\n\n"


plt.matshow(X, cmap=plt.cm.Blues)
plt.title("original dataset")

plt.matshow(X[cols], cmap=plt.cm.Blues)
plt.title("clustered dataset")
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

#f = open('gclean.data', 'w')
#pickle.dump(GClean, f)
#
#f = open("gclean.data", "r")
#GClean = pickle.load(f)
#f.close()

#GClean.add_edges_from(clean_edges)
#GDirty.add_edges_from(dirty_edges)

nx.write_gexf(GClean, "folha586.gexf")
nx.write_gexf(GDirty, "folha300dirty.gexf")

##############
#deteccao de comunidades!
import sys
sys.path.insert(0, 'louvain')
import community
import networkx as nx
import matplotlib.pyplot as plt

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
    
    
comsizes = np.bincount(partition.values())
for com in set(partition.values()):
    print "particao", com, comsizes[com]
    toptweets = np.argsort(community_matrix[com,:])[::-1][:10]
    for t in toptweets:
        print community_matrix[com, t]/comsizes[com], tweets_collection[t][0]['text'][14:] 
    print "\n"
        

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
        #tweet[users.index(follower)] += 0.4 * (1-tweet[users.index(follower)]) #valor arbitrario!
        tweet[np.where(users==follower)] = 1
#        if tweet[users.index(follower)] < 1:
#            tweet[users.index(follower)] += 1.0/len(graph.in_edges([follower]))
        #print len(visited)
 
#tweets_matrix_sparse = tweets_matrix.copy()
tweets_matrix = tweets_matrix_sparse.copy()      
for tweet in tweets_matrix.T:
    for user in np.where(tweet==1)[0]:
        #get user followers and increase their values
        propagate(users[user], GClean, tweet, [])
        
        
#tweets_matrix = tweets_matrix_sparse #restore (if needed)
        
######
#biclusterizacao

# Author: Kemal Eren <kemal@kemaleren.com>
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster.bicluster import SpectralCoclustering


#data = tweets_matrix
data = community_matrix

plt.matshow(tweets_matrix_sparse.T, cmap=plt.cm.Blues)
plt.title("Sparse dataset")

plt.matshow(tweets_matrix.T, cmap=plt.cm.Blues)
plt.title("Original dataset")

plt.matshow(data.T, cmap=plt.cm.Blues)
plt.title("Communities dataset")


model = SpectralCoclustering(n_clusters=5)
model.fit(data)


fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data.T, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.show() 



##bimax  

import bibench.all as bb
bb.heatmap(data)

import bibench.all as bb         # import most commonly used functions
data = bb.get_gds_data('GDS181') # download gene expression dataset GDS181
data = bb.pca_impute(data)       # impute missing values
biclusters = bb.plaid(data)      # cluster with Plaid algorithm

row_labels = []
column_labels = []
for bic in biclusters:
    row_labels += bic.rows
    column_labels += bic.cols
    

fit_data = data[row_labels]
fit_data = fit_data[:, column_labels]

plt.matshow(fit_data.T, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")


#imprime clusters
for bic in biclusters:
    for c in bic.cols:
        print tweets_collection[c][0]['text'] 
    print "\n\n"
    
    
    
#############################
#knn
    
#seleciona grupo para classificar
import random
from scipy.spatial.distance import cdist
from operator import itemgetter

k = 200
samplei = random.sample(xrange(tweets_matrix.shape[1]), k)
samples = tweets_matrix[:,samplei]

D = cdist(tweets_matrix.T, samples.T, metric="jaccard")


#imprime tweet sendo comparado
#for i in range(k):
#    print "original:"
#    print classes[samplei[i]], tweets_collection[samplei[i]][0]['text'][14:100]
#    
#    sorts = np.argsort(D[:,i])
#    print "10 vizinhos mais próximos:"
#    for j in range(1,10):
#        print classes[sorts[j]], "\t\t", D[sorts[j],i], "\t", tweets_collection[sorts[j]][0]['text'][14:] 
# 
#    print "\n"
    


tamanhos = [1, 3, 5, 7, 11, 25]
resultados_ok = dict([(i, 0) for i in tamanhos])
resultados_nulo = dict([(i, 0) for i in tamanhos])

for i in range(k):
    gabarito = classes[samplei[i]]
    sorts = np.argsort(D[:,i])
    shuffled = range(len(sorts))
    random.shuffle(shuffled)
    shuffled = np.array(shuffled)
    for j in range(len(tamanhos)):
        vizinhos = list(classes[sorts[1:(tamanhos[j]+1)]])
        contagem = [(vizinhos.count(v), v) for v in set(vizinhos)]
        if max(contagem, key=itemgetter(0))[1] == gabarito:
                resultados_ok[tamanhos[j]] += 1
        vizinhos = list(classes[shuffled[1:(tamanhos[j]+1)]])
        contagem = [(vizinhos.count(v), v) for v in set(vizinhos)]
        if max(contagem, key=itemgetter(0))[1] == gabarito:
                resultados_nulo[tamanhos[j]] += 1
        
k = float(k)
for i in tamanhos:
    print "k",i,": ",resultados_ok[i]/k,resultados_nulo[i]/k
k = int(k)    
 
    
    
    
    
    
    
    
    
##########################
#biclustering community matrix
