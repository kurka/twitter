# -*- coding: utf-8 -*-
import networkx as nx
import json
import numpy as np
import pickle
from PersistencyLayer import TweetsPersister
import copy

###############################################################################
#PREPARACAO DOS DADOS
###############################################################################
#1- Seleciona grupo de tweets da base

json_fp = open('credentials.json')
cred = json.load(json_fp)

persister = TweetsPersister()
#ROOTUSER = 14594813
#root_tweets = persister.loadTweetsOfUser(ROOTUSER) #get all root tweets

#load tweets already categorized by autoclassifier
f = open("folha2600classes.data", "r") 
root_tweets_extended = pickle.load(f)
f.close()


class_dict = {'cotidiano':1, 'esporte':2, 'mundo':3, 
              'poder':4, 'ilustrada':5, 'mercado':6}

def topiccleaner(tweet):
    if tweet['class'] in class_dict.keys():
        return True
    else:
        return False

#cleaning root tweets
root_tweets = filter(topiccleaner, root_tweets_extended)




tweets_collection = []
classes = []
lens = []
min_retweets = 20
for root_tweet in root_tweets:
    retweets = persister.loadRetweets(root_tweet['tweet']['tweet_id'])
    lens.append(len(retweets))
    #limiting just stories with more than min_retweets
    if retweets and len(retweets)>=min_retweets: 
        tweets_collection.append(retweets)
        classes.append(root_tweet['class'])


classes = np.array(classes)
tweets_collection = np.array(tweets_collection)
users = np.array(list(set([item['user_id'] for sublist in tweets_collection for item in sublist])))
#users.remove(ROOTUSER)



##############################################################################
#2- Monta matriz com tweets

TM_dict = {}

tweets_matrix = np.zeros((len(users), len(tweets_collection))) #empty matrix

for col_num, col in enumerate(tweets_collection):
    #get the positions of the users, in users list
    col_indexes = np.array([int(np.where(users == tweet['user_id'])[0]) for tweet in col]) 
    tweets_matrix[col_indexes, col_num] = 1
    
#remove bots from matrix
#considera quem retweetou mais que 80% dos tweets um bot
rsum = np.sum(tweets_matrix, axis=1)
bots = np.where(rsum > 0.8*len(tweets_collection))[0] 
#TODO: melhorar definicao de bots. Talvez ver por timestamp


tweets_matrix = np.delete(tweets_matrix, bots, axis=0)
users = np.delete(users, bots)



#clear empty lines or colums
lsum = np.sum(tweets_matrix, axis=0)
zl = np.where(lsum==0)[0]
tweets_matrix = np.delete(tweets_matrix, zl, axis=1)
tweets_collection = np.delete(tweets_collection, zl)
classes = np.delete(classes, zl)


rsum = np.sum(tweets_matrix, axis=1)
zr = np.where(rsum==0)[0]
tweets_matrix = np.delete(tweets_matrix, zr, axis=0)
users = np.delete(users, zr)




TM_dict['TM_sparse'] = {'tweets_matrix':tweets_matrix.copy(),
                                'users':users.copy(),
                                'classes':classes.copy(),
                                'tweets_collection':tweets_collection.copy()
                                }

#remove users with less than 5 tweets in history

rsum = np.sum(tweets_matrix, axis=1)
zr = np.where(rsum<5)[0]
tweets_matrix = np.delete(tweets_matrix, zr, axis=0)
users = np.delete(users, zr)


lsum = np.sum(tweets_matrix, axis=0)
zl = np.where(lsum==0)[0]
tweets_matrix = np.delete(tweets_matrix, zl, axis=1)
tweets_collection = np.delete(tweets_collection, zl)
classes = np.delete(classes, zl)

TM_dict['TM_pop'] = {'tweets_matrix':tweets_matrix.copy(),
                                'users':users.copy(),
                                'classes':classes.copy(),
                                'tweets_collection':tweets_collection.copy()
                                }


###############################################################################
#3- Monta grafo da rede

def buildgraph(nodes_list):
    G = nx.DiGraph()   
    edges = []
    total = float(len(nodes_list))
    for count, n in enumerate(nodes_list):
       print("%s/%s" %(count, total))
       try:   
           f = open('net/'+str(n))
       except:
           print "nao achou", n
           continue
       neighbours = json.load(f)
       for nei in neighbours:
          if nei in nodes_list: #just add if node is active
             edges.append((n,nei))
          #dirty_edges.append((n, nei))
       f.close()
       #GDirty.add_edges_from(dirty_edges)   
    G.add_edges_from(edges)
    return G

G_sparse = buildgraph(set(TM_dict['TM_sparse']['users']))
G_pop = buildgraph(set(TM_dict['TM_pop']['users']))


#tira dos TMs usuarios que não estão no grafico (porque nao tem ligacao com ninguem)
def removeIsolated(tm_dict, G):
    clean_cols = []
    for col, user in enumerate(tm_dict['users']):
        if user in G:
            clean_cols.append(col)
    
    tm_dict['tweets_matrix'] = tm_dict['tweets_matrix'][clean_cols,:]
    tm_dict['users'] = tm_dict['users'][clean_cols]
    
    #clear empty lines or colums
    lsum = np.sum(tm_dict['tweets_matrix'], axis=0)
    zl = np.where(lsum==0)[0]
    if len(zl) > 0:
        print "removendo", len(zl), "linhas"
    tm_dict['tweets_matrix'] = np.delete(tm_dict['tweets_matrix'], zl, axis=1)
    tm_dict['tweets_collection'] = np.delete(tm_dict['tweets_collection'], zl)
    tm_dict['classes'] = np.delete(tm_dict['classes'], zl)
    
    
    rsum = np.sum(tm_dict['tweets_matrix'], axis=1)
    zr = np.where(rsum==0)[0]
    if len(zr) > 0:
        print "removendo", len(zr), "colunas"
    tm_dict['tweets_matrix'] = np.delete(tm_dict['tweets_matrix'], zr, axis=0)
    tm_dict['users'] = np.delete(tm_dict['users'], zr)
    
    
removeIsolated(TM_dict['TM_sparse'], G_sparse)
removeIsolated(TM_dict['TM_pop'], G_pop)

###############################################################################
#4- propagacao!

TM_dict['TM_sparse_prop'] = copy.deepcopy(TM_dict['TM_sparse'])
TM_dict['TM_pop_prop'] = copy.deepcopy(TM_dict['TM_pop'])
    
def propagate(users, user, graph, tweet, visited):
    #visited.append(user)    
    if user in graph:
        for follower in graph.successors(user):
            #if follower not in visited:
                #propagate(follower, graph, tweet, visited)    
            #tweet[users.index(follower)] += 0.4 * (1-tweet[users.index(follower)]) #valor arbitrario!            
            tweet[np.where(users==follower)] = 1
            #if tweet[np.where(users==follower)] < 1:
            #    tweet[np.where(users==follower)] += 1.0/len(graph.in_edges([follower]))
            #print len(visited)

def propagate_tweetsmatrix(tm_dict, G):
  
    print sum(tm_dict['tweets_matrix'])  
    for tweet in tm_dict['tweets_matrix'].T:
        for user in np.where(tweet==1)[0]:
            #get user followers and increase their values
            propagate(tm_dict['users'], tm_dict['users'][user], G, tweet, [])
    print sum(tm_dict['tweets_matrix'])
    
    return tm_dict['tweets_matrix']
   

TM_dict['TM_sparse_prop']['tweets_matrix'] = propagate_tweetsmatrix(TM_dict['TM_sparse_prop'], G_sparse)
TM_dict['TM_pop_prop']['tweets_matrix'] = propagate_tweetsmatrix(TM_dict['TM_pop_prop'], G_pop)

###############################################################################
#deteccao de comunidades
import sys
sys.path.insert(0, 'louvain')
import community


TM_dict['TM_sparse_com'] = copy.deepcopy(TM_dict['TM_sparse'])
TM_dict['TM_pop_com'] = copy.deepcopy(TM_dict['TM_pop'])

def community_matrix(tm_dict, G):
    
    tweets_matrix = tm_dict['tweets_matrix']    
    G_un = G.to_undirected()
    #first compute the best partition
    partition = community.best_partition(G_un)
    
    
    #agrupar as linhas!
    community_matrix = np.zeros((len(set(partition.values())), len(tm_dict['tweets_collection']))) #empty matrix
    
    
    #jeito 1: soma os valores de toda a comunidade
    row_num = 0
    for com in set(partition.values()):
        list_nodes = [node for node in partition.keys()
                                    if partition[node] == com]
        row_indexes = np.array([int(np.where(tm_dict['users'] == user)[0]) for user in list_nodes]) 
        print len(list_nodes) #get the size of the list of nodes
        community_matrix[row_num, :] = tweets_matrix[row_indexes, :].sum(0)
        row_num += 1
    
    #similaridade entre comunidades  
    comsizes = np.bincount(partition.values())
    for com in set(partition.values()):
        toptweets = np.argsort(community_matrix[com,:])[::-1][:10]
        if comsizes[com] > 10:
            print "particao", com, comsizes[com]
            for t in toptweets:
                if community_matrix[com,t] > 0:
                    print community_matrix[com, t]/comsizes[com], \
                          sum(tweets_matrix[:,t])/float(tweets_matrix.shape[0]), \
                          tm_dict['tweets_collection'][t][0]['text'][14:] 
            print "\n"
            
    return community_matrix, partition


TM_dict['TM_sparse_com']['tweets_matrix'], TM_dict['TM_sparse_com']['partitions'] = community_matrix(TM_dict['TM_sparse_prop'], G_sparse)
TM_dict['TM_pop_com']['tweets_matrix'], TM_dict['TM_pop_com']['partitions']  = community_matrix(TM_dict['TM_pop_prop'], G_pop)


#salva tudo


f = open("processed.data", "w")
pickle.dump((TM_dict, G_sparse, G_pop), f)
f.close()


###############################################################################
###############################################################################
#Machine Learning
###############################################################################
###############################################################################


import pickle
f = open("processed.data", "r")
(TM_dict, G_sparse, G_pop) = pickle.load(f)
f.close()

############################################


import random
from scipy.spatial.distance import cdist
from operator import itemgetter
import matplotlib.pyplot as plt

#KNN
    
#seleciona grupo para classificar


n = 300 #number of samples
samplei = random.sample(xrange(TM_dict['TM_pop']['tweets_matrix'].shape[1]), n) #select a group of tweets ids as sample

samples = {}
keys = TM_dict.keys()

for key in keys:
    samples[key] = TM_dict[key]['tweets_matrix'][:,samplei]
D = {}
for key in keys:
    D[key] = cdist(TM_dict[key]['tweets_matrix'].T, samples[key].T, metric="jaccard")


#imprime tweet sendo comparado
#for i in range(n):
#    print "original:"
#    print classes[samplei[i]], tweets_collection[samplei[i]][0]['text'][14:100]
#    
#    sorts = np.argsort(D[:,i])
#    print "10 vizinhos mais próximos:"
#    for j in range(1,10):
#        print classes[sorts[j]], "\t\t", D[sorts[j],i], "\t", tweets_collection[sorts[j]][0]['text'][14:] 
# 
#    print "\n"    


tamanhos = [1, 3, 5, 7, 11, 25, 40, 50]
resultados = np.zeros((len(TM_dict)+1, len(tamanhos)))

for i in range(n):

    #aplica knn para cada matriz
    for j, key in enumerate(keys):
        if i == 0:
            print j, key
        classes = TM_dict[key]['classes']
        gabarito = classes[samplei[i]]
        vizinhos_sort = np.argsort(D[key][:,i]) #vizinhos ordenados por distancia
        for k in range(len(tamanhos)):
            kvizinhos = list(classes[vizinhos_sort[1:(tamanhos[k]+1)]])
            contagem = [(kvizinhos.count(v), v) for v in set(kvizinhos)]
            if max(contagem, key=itemgetter(0))[1] == gabarito:
                    resultados[j,k] += 1
    
    #faz teste nulo, para comparacao
    classes = TM_dict['TM_sparse']['classes']
    gabarito = classes[samplei[i]]         
    vizinhos_rand = np.random.permutation(vizinhos_sort)
    for k in range(len(tamanhos)):
        kvizinhos = list(classes[vizinhos_rand[1:(tamanhos[k]+1)]])
        contagem = [(kvizinhos.count(v), v) for v in set(kvizinhos)]
        if max(contagem, key=itemgetter(0))[1] == gabarito:
                resultados[-1,k] += 1    

print resultados    
        
for i, tam in enumerate(tamanhos):
    print "K =", tam
    for j, nome in enumerate(keys):
        print nome, ":\t", resultados[j,i]/float(n)
        
    print "random :", resultados[-1, i]/float(n)
    print   


###############################################################################
#regressao linear e SVM
from sklearn import svm

def prepare_sets(tm_dict, split=0.5):
    
    class_dict = {'cotidiano':1, 'esporte':2, 'mundo':3, 
              'poder':4, 'ilustrada':5, 'mercado':6}    
    Y = np.array([class_dict[classe] for classe in tm_dict['classes']])
#    Y_rand = list(Y)
#    random.shuffle(Y_rand)
#    Y_rand = np.array(Y_rand) #TODO: comparar com nulo
    
    X = tm_dict['tweets_matrix'].T
    
    
    total_size = Y.size
    random_indices = np.random.permutation(total_size)
    
    
    train_limit = split
    Y_train = Y[random_indices[:int(train_limit * total_size)]]
    X_train = X[random_indices[:int(train_limit * total_size)]]
    
    Y_test = Y[random_indices[int(train_limit * total_size):]]
    X_test = X[random_indices[int(train_limit * total_size):]]
    
    return (X_train, Y_train, X_test, Y_test)

def linear_reg(tm_dict):
    (X_train, Y_train, X_test, Y_test) = prepare_sets(tm_dict)
    
    teta = np.dot(np.linalg.pinv(X_train), Y_train)

    Y_l = np.around(np.dot(X_train, teta))
    print "acertos treino:", sum(Y_train == Y_l) / float(len(Y_l))
        
    Y_l = np.around(np.dot(X_test, teta))    
    print "acertos teste:", sum(Y_test == Y_l) / float(len(Y_l))



#svm
def svc(tm_dict):
    (X_train, Y_train, X_test, Y_test) = prepare_sets(tm_dict, split=0.8)
    classificador = svm.SVC(kernel='linear')
    classificador.fit(X_train, Y_train)
    
    Y_class = classificador.predict(X_train)
    acertos = sum(Y_train == Y_class) / float(len(Y_class))
    print "acertos treino:", acertos 

   
    Y_classificado = classificador.predict(X_test)
    acertos = sum(Y_test == Y_classificado) / float(len(Y_classificado))
    print "acertos teste:", acertos
    

for (key,item) in TM_dict.iteritems():
    print key
    svc(item)


#TODO: redes neurais
#TODO: LDA



###############################################################################
#CLUSTERING
import random
from kurkameans import kmeans
#kmeans varias distancias

ncluster = 20

for key in TM_dict.keys():  
    print key
    X = TM_dict[key]['tweets_matrix'].T
  
    initcentres = X[random.sample(xrange(X.shape[0]), ncluster)]
    centres, xtoc, dist = kmeans(X, ncluster, initcentres=None, metric="cosine", maxiter=20, delta=.00001, verbose=True)

   
   
tweets_collection = TM_dict[key]['tweets_collection']   
#####
#imprime clusters
cols = []
for j in range(ncluster):
    for i in np.where(xtoc == j)[0]:
        print classes[i], "\t", tweets_collection[i][0]['text'][14:] 
        cols.append(i)
    print "\n\n"


#plt.matshow(X, cmap=plt.cm.Blues)
#plt.title("original dataset")
#
#plt.matshow(X[cols], cmap=plt.cm.Blues)
#plt.title("clustered dataset")

        
###############################################################################
#PCA

from sklearn.decomposition import PCA

X = TM_dict['TM_sparse']['tweets_matrix'].T
class_dict = {'cotidiano':1, 'esporte':2, 'mundo':3, 
          'poder':4, 'ilustrada':5, 'mercado':6}    
y = np.array([class_dict[classe] for classe in TM_dict['TM_sparse']['classes']])


pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)


plt.scatter(X[:,0], X[:,1], c=y)

        
###############################################################################
#biclusterizacao (inclose)

import scipy.io as sio
import os

#inclose

TM_dict['TM_pop']['partitions'] = TM_dict['TM_pop_com']['partitions']
TM_dict['TM_sparse']['partitions'] = TM_dict['TM_sparse_com']['partitions']

#for tm_dict in [TM_dict['TM_sparse'], TM_dict['TM_pop']]:
for tm_dict in [TM_dict['TM_pop']]:
    #1-roda inclose usando octave
    sio.savemat('matlab/twitter.mat', {'tweets_matrix':tm_dict['tweets_matrix']})
    print "executando inclose. Isso pode demorar um pouco"
    os.system("octave twitter-inclose.m")
    mat_contents = sio.loadmat('matlab/bicl.mat')
    biclusters = mat_contents['biclusters'][0]
    
    
    #2 imprime biclusters encontrados
    tweets_collection = tm_dict['tweets_collection']
    classes = tm_dict['classes']

    #3 encontra comunidades que os elementos do bicluster fazem parte
    community_users = {}
    for com in set(tm_dict['partitions'].values()):
        list_nodes = [node for node in tm_dict['partitions'].keys()
                                    if tm_dict['partitions'][node] == com]
        row_indexes = np.array([int(np.where(tm_dict['users'] == user)[0]) for user in list_nodes]) 
        community_users[com] = row_indexes


    bic2com = []
    for bic in biclusters:
        bic_users = bic['B'][0]
        nearest_com = -1
        biggest_similarity = 0
        #find community that contains most of the bicluster's users
        for com in set(tm_dict['partitions'].values()):
            similars = len(set(bic_users) & set(community_users[com]))
            if similars > biggest_similarity:
                nearest_com = com
                biggest_similarity = similars
        print "bic size:", len(bic_users), "/ com size:", len(community_users[nearest_com]),\
              "/ in common:", biggest_similarity
                
        bic2com.append(nearest_com)
    
    print bic2com
            
        
#    for bic in biclusters:
#        print "usuarios:", len(bic['A'][0]), "tweets:", len(bic['B'][0])
#        for c in bic['B'][0]:
#            i = int(c-1)
#            print classes[i], "\t", tweets_collection[i][0]['text'][14:] 
#        print "\n"  
#    for bic in biclusters:
#        print "usuarios:", len(bic['A'][0]), "tweets:", len(bic['B'][0])
#        for c in bic['B'][0]:
#            i = int(c-1)
#            print classes[i], "\t", tweets_collection[i][0]['text'][14:] 
#        print "\n"


#TODO: ver similaridades entre comunidades e biclusters
TM_dict['TM_pop_com']['partitions']

###############################################################################
#ESTATISTICAS
###############################################################################
from scipy.spatial.distance import cdist
import random



#distancia grupos X distancia global


#global distance:

X = tweets_matrix.T

g_centroid = np.mean(X, axis=0)
g_centroid.shape = (1, X.shape[1])
g_dists = cdist(X, g_centroid, metric='euclidean')
g_dist = np.mean(g_dists)
print "Global dist:", g_dist

for cat in class_dict.keys():
    topics_id = np.where(classes==cat)
    topic_centroid = np.mean(X[topics_id], axis=0)
    topic_centroid.shape = (1, X.shape[1])
    t_dists = cdist(X[topics_id], topic_centroid, metric='euclidean')
    t_dist = np.mean(t_dists)
    print cat, "dist:", t_dist
    
    random_id = random.sample(xrange(tweets_matrix.shape[1]), len(topics_id))
    random_centroid = np.mean(X[random_id], axis=0)
    random_centroid.shape = (1, X.shape[1])
    t_dists = cdist(X[random_id], random_centroid, metric='euclidean')
    t_dist = np.mean(t_dists)
    print "random dist:", t_dist
    
    

#TODO: comparar com grupos aleatorios
#TODO: PCA

###############################################################################
#topicos diferentes por usuario


#considera apenas usuarios com 6 ou mais tweets
tweetssum = np.sum(TM_dict['TM_sparse']['tweets_matrix'], axis=1)
topusers = np.where(tweetssum>=6)[0]


bcount = []
for usr in TM_dict['TM_sparse']['tweets_matrix'][topusers]:
    usr_topics = set(TM_dict['TM_sparse']['classes'][np.where(usr==1)])
    print len(usr_topics), usr_topics
    bcount.append(len(usr_topics))
    
np.bincount(bcount)

    
###############################################################################
#frequencia dentro de comunidades
    
for tm_dict in [TM_dict['TM_sparse_com']]:
    
    community_matrix = tm_dict['tweets_matrix']
    partition = tm_dict['partitions']
    tweets_matrix = TM_dict['TM_sparse']['tweets_matrix']
    
    #similaridade entre comunidades  
    comsizes = np.bincount(partition.values())
    for com in set(partition.values()):
        toptweets = np.argsort(community_matrix[com,:])[::-1][:3]
        if comsizes[com] > 100:
            print "\multicolumn{3}{l}{Comunidade %d, membros: %d, total tweets: %d} \\\\\n"\
                    % (com, comsizes[com], sum(community_matrix[com,:]))
            for t in toptweets:
                if community_matrix[com,t] > 0:
                    print "%2.1f\\%% & %2.1f\\%% & %s...\\\\\n"\
                          %(100*community_matrix[com, t]/comsizes[com], \
                          100*sum(tweets_matrix[:,t])/float(tweets_matrix.shape[0]), \
                          #100*community_matrix[com, t]/float(sum(tweets_matrix[:,t])), \
                          tm_dict['tweets_collection'][t][0]['text'][14:70]) 
                          
            print "\hline\n"

#TODO:
#LDA?
#redes neurais


\multicolumn{4}{l}{Comunidade 1, tamanho: 10, tweets: 20} \\
35 & 54 & 40 & Neymar pode ser preso se mostrar a cueca com




#obsoletos

#from sklearn.cluster.bicluster import SpectralCoclustering
#
##data = community_matrix
#data = TM_dict['TM_pop']['tweets_matrix'].
#plt.matshow(TM_dict['TM_pop']['tweets_matrix'].T, cmap=plt.cm.Blues)
#plt.title("Sparse dataset")
#
#plt.matshow(TM_dict['TM_pop_prop']['tweets_matrix'].T, cmap=plt.cm.Blues)
#plt.title("Original dataset")
#
#plt.matshow(data.T, cmap=plt.cm.Blues)
#plt.title("Communities dataset")
#
#
#model = SpectralCoclustering(n_clusters=5)
#model.fit(data)
#
#
#fit_data = data[np.argsort(model.row_labels_)]
#fit_data = fit_data[:, np.argsort(model.column_labels_)]
#
#plt.matshow(fit_data.T, cmap=plt.cm.Blues)
#plt.title("After biclustering; rearranged to show biclusters")
#
#plt.show() 

##bimax  

#import bibench.all as bb
#bb.heatmap(data)
#
#import bibench.all as bb         # import most commonly used functions
#data = bb.get_gds_data('GDS181') # download gene expression dataset GDS181
#data = bb.pca_impute(data)       # impute missing values
#biclusters = bb.plaid(data)      # cluster with Plaid algorithm
#
#row_labels = []
#column_labels = []
#for bic in biclusters:
#    row_labels += bic.rows
#    column_labels += bic.cols
#    
#
#fit_data = data[row_labels]
#fit_data = fit_data[:, column_labels]
#
#plt.matshow(fit_data.T, cmap=plt.cm.Blues)
#plt.title("After biclustering; rearranged to show biclusters")
#
#
##imprime clusters
#for bic in biclusters:
#    for c in bic.cols:
#        print tweets_collection[c][0]['text'] 
#    print "\n\n" 