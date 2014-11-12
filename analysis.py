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

FOLHA = '14594813'
ESTADAO = '9317502'
UOLNOT = '14594698'
G1 = '8802752'
R7 = '65473559'

ROOTUSERS = [int(FOLHA), int(ESTADAO), int(UOLNOT), int(G1), int(R7)]

#ROOTUSER = 14594813
#root_tweets = persister.loadTweetsOfUser(ROOTUSER) #get all root tweets


root_tweets = []
for root in ROOTUSERS:
    rt = persister.loadTweetsOfUser(root)
    print len(rt)
    root_tweets.extend(rt[:-100])
    
print len(root_tweets)

##load tweets already categorized by autoclassifier
#f = open("folha2600classes.data", "r") 
#root_tweets_extended = pickle.load(f)
#f.close()


#class_dict = {'cotidiano':0, 'esporte':1, 'mundo':2, 
#              'poder':3, 'ilustrada':4, 'mercado':5}
#
#def topiccleaner(tweet):
#    if tweet['class'] in class_dict.keys():
#        return True
#    else:
#        return False
#
##cleaning root tweets
#root_tweets = filter(topiccleaner, root_tweets_extended)




tweets_collection = []
#classes = []
lens = []
min_retweets = 50
for root_tweet in root_tweets:
    retweets = persister.loadRetweets(root_tweet['tweet_id'])
    lens.append(len(retweets))
    #limiting just stories with more than min_retweets
    if retweets and len(retweets)>=min_retweets: 
        tweets_collection.append(retweets)
        #classes.append(root_tweet['class'])


#classes = np.array(classes)
tweets_collection = np.array(tweets_collection)
users = np.array(list(set([item['user_id'] for sublist in tweets_collection for item in sublist])))
#users.remove(ROOTUSER)

f = open("tweetscol0825.data", "w")
pickle.dump( tweets_collection, f)
f.close()


##############################################################################
#2- Monta matriz com tweets

TM_dict = {}

tweets_matrix = np.zeros((len(users), len(tweets_collection)), dtype='int8') #empty matrix

for col_num, col in enumerate(tweets_collection):
    #get the positions of the users, in users list
    col_indexes = np.array([int(np.where(users == tweet['user_id'])[0]) for tweet in col]) 
    tweets_matrix[col_indexes, col_num] = 1



f = open("twetsmatrix0825.data", "w")
pickle.dump( tweets_matrix, f)
f.close()

#remove bots from matrix
#considera quem retweetou mais que 80% dos tweets um bot
rsum = np.sum(tweets_matrix, axis=1)
bots = np.where(rsum > 0.6*len(tweets_collection))[0] 
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
                                #'classes':classes.copy(),
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
                                #'classes':classes.copy(),
                                'tweets_collection':tweets_collection.copy()
                                }


###############################################################################
#3- Monta grafo da rede

def buildgraph(nodes_list):
    G = nx.DiGraph()   
    edges = []
    total = float(len(nodes_list))
    for count, n in enumerate(nodes_list):
       print("%s/%s %s" %(count, total, n))
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
    #tm_dict['classes'] = np.delete(tm_dict['classes'], zl)
    
    
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
import copy
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
        print len(list_nodes) #get the community size
        community_matrix[row_num, :] = tweets_matrix[row_indexes, :].sum(0)#/float(len(row_indexes))
        row_num += 1
    
#    #similaridade entre comunidades  
#    comsizes = np.bincount(partition.values())
#    for com in set(partition.values()):
#        toptweets = np.argsort(community_matrix[com,:])[::-1][:5]
#        if comsizes[com] > 10:
#            print "particao", com, comsizes[com]
#            for t in toptweets:
#                if community_matrix[com,t] > 0:
#                    print community_matrix[com, t]/comsizes[com], \
#                          sum(tweets_matrix[:,t])/float(tweets_matrix.shape[0]), \
#                          tm_dict['tweets_collection'][t][0]['text'][14:] 
#            print "\n"
            
    return community_matrix, partition


TM_dict['TM_sparse_com']['tweets_matrix'], TM_dict['TM_sparse_com']['partitions'] = community_matrix(TM_dict['TM_sparse'], G_sparse)
TM_dict['TM_pop_com']['tweets_matrix'], TM_dict['TM_pop_com']['partitions']  = community_matrix(TM_dict['TM_pop'], G_pop)



#salva tudo
f = open("Graphs0828.data", "w")
pickle.dump((G_sparse, G_pop), f)
f.close()

f = open("TMdict0828.data", "w")
pickle.dump(TM_dict, f)
f.close()


###############################################################################
###############################################################################
#Machine Learning
###############################################################################
##############################################################################


import pickle
f = open("processed.data", "r")
(TM_dict, G_sparse, G_pop) = pickle.load(f)
f.close()

#TM_dict = np.load("TMdict0828.npy").item()

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

res_sparse = 100*resultados[keys.index('TM_sparse')]/float(n)
res_random = 100*resultados[-1]/float(n)

width = 0.35
b1 = plt.bar(np.arange(len(tamanhos)), res_sparse, width, color='y')
b2 = plt.bar(np.arange(len(tamanhos))+width, res_random, width, color='r')
plt.xticks(np.arange(len(tamanhos))+width, tamanhos)
plt.legend( (b1[0], b2[0]), ('KNN', 'null-model') )
plt.ylabel("porcentagem de acertos")
plt.xlabel("k")
plt.title("Classificacao KNN")
plt.ylim((0,50))





###############################################################################
#regressao linear e SVM
from sklearn import svm

def prepare_sets(tm_dict, split=0.5):
    
    class_dict = {'cotidiano':0, 'esporte':1, 'mundo':2, 
              'poder':3, 'ilustrada':4, 'mercado':5}    
    Y = np.array([class_dict[classe] for classe in tm_dict['classes']])
    Y.shape = (Y.size, 1) #create column vector
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
    acertos = sum(Y_train == Y_class) / float(Y_class.size)
    print "acertos treino:", acertos 

   
    Y_classificado = classificador.predict(X_test)
    acertos = sum(Y_test == Y_classificado) / float(Y_classificado.size)
    print "acertos teste:", acertos
    

for (key,item) in TM_dict.iteritems():
    print key
    svc(item)



###############################################################################
#MLP

#ver http://pybrain.org/docs/tutorial/fnn.html
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer


#means = [(-1,0),(2,4),(3,1)]
#cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
#alldata = ClassificationDataSet(2, 1, nb_classes=3)
#for n in xrange(400):
#    for klass in range(3):
#        input = multivariate_normal(means[klass],cov[klass])
#        alldata.addSample(input, [klass])
#
#tstdata,trndata = alldata.splitWithProportion( 0.25 )        

(X_train, Y_train, X_test, Y_test) = prepare_sets(TM_dict['TM_sparse'], split=1.0)
class_dict = {'cotidiano':0, 'esporte':1, 'mundo':2, 
              'poder':3, 'ilustrada':4, 'mercado':5}
ord_labels = ['cotidiano', 'esporte', 'mundo', 'poder', 'ilustrada', 'mercado']
              
DS =  ClassificationDataSet(inp=X_train.shape[1], nb_classes=len(class_dict), class_labels=ord_labels)
assert(X_train.shape[0] == Y_train.shape[0])
DS.setField('input', X_train)
DS.setField('target', Y_train)

tstdata,trndata = DS.splitWithProportion( 0.25 )

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

nneuronios = 10
fnn = buildNetwork( trndata.indim, nneuronios, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)


epochs = []
trn_errors = []
tst_errors = []
best_result = 100
for _ in range(40):
    trainer.trainEpochs(5)
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
          
    if tstresult < best_result:
        best_result = tstresult
        print "best"

    epochs.append(trainer.totalepochs)
    trn_errors.append(100-trnresult)
    tst_errors.append(100-tstresult)    
    
    
plt.plot(epochs, trn_errors, epochs, tst_errors)
plt.xlabel('Epocas')
plt.ylabel('Acertos (%)') 
plt.legend(("Treinamento","Teste"))
plt.title("Rede Neural MLP, %s neuronios, 1 camada escondida" %nneuronios )
   



###############################################################################
#CLUSTERING
import random
from kurkameans import kmeans
#kmeans varias distancias

ncluster = 20

#for key in TM_dict.keys():  
for key in ["TM_pop"]:
    print key
    X = TM_dict[key]['tweets_matrix'].T
  
    initcentres = X[random.sample(xrange(X.shape[0]), ncluster)]
    centres, xtoc, dist = kmeans(X, ncluster, initcentres=None, metric="cosine", maxiter=20, delta=.00001, verbose=True)

   
   
    tweets_collection = TM_dict[key]['tweets_collection']   
    #####
    #imprime clusters
    cols = []
    for j in range(ncluster):
        members = np.where(xtoc == j)[0]
        if len(members) >= 5 and len(members) <= 10: #optional limit
            for i in np.where(xtoc == j)[0]:
                #print classes[i], "\t", tweets_collection[i][0]['text'][14:]
                print tweets_collection[i][0]['text'][15:], "\\\\"
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


pca = PCA()#n_components)
pca.fit(X)
X2 = pca.transform(X)


X2 = np.delete(X2, 178, axis=0)
y = np.delete(y, 178)
X2 = np.delete(X2, 134, axis=0)
y = np.delete(y, 134)

plt.scatter(X2[:,0], X2[:,1], c=y)

        
###############################################################################
#biclusterizacao (inclose)

import scipy.io as sio
import os

#inclose

def inclose(tmatrix, minrow, mincol):
    #Roda inclose usando octave
    sio.savemat('matlab/twitter.mat', {'tweets_matrix':tmatrix, 'minrow':minrow, 'mincol':mincol})
    os.chdir("matlab")
    os.system("octave twitter-inclose.m")
    os.chdir("..")
    os.system("cd ..")
    mat_contents = sio.loadmat('matlab/bicl.mat')
    biclusters = mat_contents['biclusters'][0]
    return biclusters

    

tmkeys = ['TM_sparse', 'TM_pop']#TM_dict.keys()

biclusters = {}
for k in tmkeys:
    biclusters[k] = inclose(TM_dict[k]['tweets_matrix'], 3, 3)


#imprime mensagem e topicos dos biclusters
for k in tmkeys:
    print k
    for n, bic in enumerate(biclusters[k]):
        print "bic %s" %n
        bic_msgs = bic['B'][0]-1
        for msg in bic_msgs:
            print TM_dict[k]['classes'][msg], "\t", TM_dict[k]['tweets_collection'][msg][0]['text'][14:]
        print "\n"
        
#verifica correlacao entre bicluster e grupos
TM_dict['TM_pop']['partitions'] = TM_dict['TM_pop_com']['partitions']
TM_dict['TM_sparse']['partitions'] = TM_dict['TM_sparse_com']['partitions']

for key in ['TM_pop']:
    #1-roda inclose usando octave
    bicluster = biclusters[key]

 
        
    community_users = {}
    for com in set(TM_dict[key]['partitions'].values()):
        list_nodes = [node for node in TM_dict[key]['partitions'].keys()
                                    if TM_dict[key]['partitions'][node] == com]
        row_indexes = np.array([int(np.where(TM_dict[key]['users'] == user)[0]) for user in list_nodes]) 
        print len(row_indexes)
        community_users[com] = row_indexes

    #3 encontra comunidades que os elementos do bicluster fazem parte 
    bic2com = []
    for bic in bicluster:
        bic_users = bic['A'][0]-1
        nearest_com = -1
        biggest_similarity = 0
        #find community that contains most of the bicluster's users
        for com in set(TM_dict[key]['partitions'].values()):
            similars = len(set(bic_users) & set(community_users[com]))
            if similars > biggest_similarity:
                nearest_com = com
                biggest_similarity = similars
        print "bic size:", len(bic_users), "/ com size:", len(community_users[nearest_com]),\
              "/ in common:", biggest_similarity, "/ percentage:", 100*biggest_similarity/float(len(bic_users))
                
        bic2com.append(nearest_com)

#verifica se existem ligacoes entre usuarios de um mesmo bicluster    
TM_dict['TM_pop']['graph'] = G_pop
TM_dict['TM_sparse']['graph'] = G_sparse



from itertools import combinations
def check_connectivity(G, nodes_list):
    n_nodes = len(nodes_list)
    n_combinations = ((n_nodes-1)*n_nodes)/2
    
    n_edges = 0
    for u,v in combinations(nodes_list, 2):
        if G.has_edge(u,v) or G.has_edge(v,u):
            n_edges += 1
    return 100*n_edges / float(n_combinations)


  
for key in ['TM_pop']:    
    bic2com = []
    for bic in biclusters[key]:
        bic_users = bic['A'][0]-1
        connectivity = check_connectivity(TM_dict[key]['graph'], TM_dict[key]['users'][bic_users.astype(int)])
        print "bic size:", len(bic_users), "/ connectivity:", connectivity
        
            
        
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
    
distr = 100*np.bincount(bcount)/float(topusers.size)
bars = distr[1:]

plt.bar(np.arange(len(bars)), bars, align='center')
plt.xticks(np.arange(6), ['1','2','3','4','5','6'])
plt.ylabel("porcentagem dos usuarios")
plt.xlabel("numero de categorias compartilhadas")
plt.title("Quantidade de topicos compartilhados por usuario")

###############################################################################
#histogramas

#tweets por usuario
twpusr = np.sum(TM_dict['TM_sparse']['tweets_matrix'], axis=1)
plt.hist(np.sort(twpusr), bins=np.sort(twpusr)[-1], log=True)
plt.ylabel('Frequencia (log)')
plt.xlabel('Compartilhamentos por usuario')
plt.title("Histograma: retweets por usuario")



#log-log
bins = np.arange(0, 2.1, 0.2)
plt.xticks(bins, ["%.0f" %10**i if i % .5 == 0 else '' for i in bins])
plt.hist(np.log10(twpusr), log=True, bins=bins)
plt.ylabel('Frequencia (log)')
plt.xlabel('Compartilhamentos por usuario (log)')
plt.title("Histograma: retweets por usuario (logXlog)")


#retweets por mensagem
rtpmsg = np.sum(TM_dict['TM_sparse']['tweets_matrix'], axis=0)
plt.hist(np.sort(rtpmsg), bins=np.sort(rtpmsg)[-1], log=True)
plt.ylabel('Frequencia (log)')
plt.xlabel('Compartilhamentos por mensagem')
plt.title("Histograma: retweets por mensagem")

bins = np.arange(1, 3.1, 0.1)
plt.xticks(bins, ["%.0f" %10**i if round(10**i) in [10, 20, 30, 50, 100, 200, 300, 1000] else '' for i in bins])
#plt.xticks(bins, ["%.0f" %10**i if np.round(10**i)% 10 == 0 else '' for i in bins])
#plt.xticks(bins, ["%.0f" %10**i for i in bins])
plt.hist(np.log10(rtpmsg), log=True, bins=bins)
plt.ylabel('Frequencia (log)')
plt.xlabel('Compartilhamentos por mensagem (log)')
plt.title("Histograma: retweets por mensagem (logXlog)")
    
###############################################################################
#frequencia dentro de comunidades

for tm_dict in [TM_dict['TM_sparse_com']]:
    
    community_matrix = tm_dict['tweets_matrix']
    partition = tm_dict['partitions']
    tweets_matrix = TM_dict['TM_sparse']['tweets_matrix']
    
    comorder = []
    comorder2 = []
    
    #similaridade entre comunidades  
    comsizes = np.bincount(partition.values())
    for com in printorder:#set(partition.values()): #printorder:
        difftweets = np.absolute(community_matrix[com, :]/float(comsizes[com]) - \
                    (tweets_matrix.sum(0)-community_matrix[com, :])/float(tweets_matrix.shape[0]-comsizes[com]) )
        
        topdifftweets = np.argsort(difftweets)[::-1][:5]
        
        if comsizes[com] >= 50:#> 50 and comsizes[com] <= 100:
            comorder.append(sum(difftweets))
            comorder2.append(com)
            print "\multicolumn{3}{l}{Comunidade %d, membros: %d, total tweets: %d} \\\\\n"\
                    % (com, comsizes[com], sum(community_matrix[com,:]))
            for t in topdifftweets:
                #if community_matrix[com,t] > 0:
                print "%2.1f\\%% & %2.1f\\%% & %s...\\\\\n"\
                      %(100*community_matrix[com, t]/comsizes[com], \
                      100*(sum(tweets_matrix[:,t])-community_matrix[com, t])/float(tweets_matrix.shape[0]-comsizes[com]), \
                      #100*community_matrix[com, t]/float(sum(tweets_matrix[:,t])), \
                      tm_dict['tweets_collection'][t][0]['text'][14:80]) 
                          
            print "\hline\n"
            
    ordercom = np.argsort(comorder)[::-1]
    printorder = np.array(comorder2)[ordercom]


#bar charts dos topicos por comunidade
for tm_dict in [TM_dict['TM_sparse_com']]:
    
    community_matrix = tm_dict['tweets_matrix']
    partition = tm_dict['partitions']
    tweets_matrix = TM_dict['TM_sparse']['tweets_matrix']

    #ordercom = np.argsort(comorder)[::-1]
    #printorder = np.array(comorder2)[ordercom]
    #printorder = [17, 28,  1, 11, 13, 24,  8, 33,  3, 10,  6,  5,  7,  4,  2,  0]

    class_dict = {'cotidiano':0, 'esporte':1, 'mundo':2, 
              'poder':3, 'ilustrada':4, 'mercado':5} 
            
    for num, com in [(0,24)]:#enumerate(printorder):#[:5]):
        bars = np.array([0,0,0,0,0,0])
        for cel_pos, cel_val in enumerate(community_matrix[com,:]):
            classe = class_dict[tm_dict['classes'][cel_pos]]
            bars[classe] += cel_val
        bars = (bars / float(sum(community_matrix[com,:])))*100
        print com
        print bars
        
        plt.subplot(1,2,num+1)
        plt.bar(np.arange(len(bars)), bars, align='center', color=['b','g','r','c','m','y'], label=class_dict.keys())
        plt.xticks(np.arange(6), class_dict.keys(), rotation=90) #FIXME: ver se keys() segue a ordem de bars
        plt.ylabel("porcentagem")
        plt.title("Distribuicao Topicos Comunidade %s" %com)
        #plt.savefig("bar%s" %num)
        #plt.close()
   
        
    #geral
    num_classes = []
    bars = np.array([0,0,0,0,0,0])
    for cel_pos, cel_val in enumerate(np.sum(TM_dict['TM_sparse']['tweets_matrix'], axis=0)):
        classe = class_dict[TM_dict['TM_sparse']['classes'][cel_pos]]
        bars[classe] += cel_val
    bars = (bars / float(sum(TM_dict['TM_sparse']['tweets_matrix'])))*100
    print com
    print bars
    plt.subplot(1,2,2)
    plt.bar(np.arange(len(bars)), bars, align='center', color=['b','g','r','c','m','y'], label=class_dict.keys())
    plt.xticks(np.arange(6), class_dict.keys(), rotation=90)
    plt.ylabel("porcentagem")
    plt.title("Distribuicao Topicos Geral")
    plt.savefig("bargeral")
    plt.tight_layout()

    


###############################################################################
#análise de grafos
import networkx as nx

in_degrees = G_sparse.in_degree() # dictionary node:degree
in_values = sorted(set(in_degrees.values())) 
in_hist = [in_degrees.values().count(x) for x in in_values]

out_degrees = G_sparse.out_degree() # dictionary node:degree
out_values = sorted(set(out_degrees.values())) 
out_hist = [out_degrees.values().count(x) for x in out_values]

plt.figure()
plt.plot(in_values,in_hist,'ro') # in-degree
plt.plot(out_values,out_hist,'bv') # out-degree
plt.yscale('log')
plt.xscale('log')
plt.legend(['Seguidores','Seguidos'])
plt.xlabel('Grau (log)')
plt.ylabel('Numero de nos (log)')
plt.title('Rede de conexoes entre usuarios')
plt.savefig('graphinouthist.pdf')
plt.show()
plt.close()



G_sparse_components = nx.connected_component_subgraphs(G_sparse.to_undirected())



# Betweenness centrality
bet_cen = nx.betweenness_centrality(G_sparse)
# Closeness centrality
clo_cen = nx.closeness_centrality(G_sparse)
# Eigenvector centrality
eig_cen = nx.eigenvector_centrality(G_sparse)
#  Average clustering
avg_clus = nx.average_clustering(G_sparse.to_undirected()) 
# Average clustering of random network:
m = len(G_sparse.to_undirected().edges())
n = len(G_sparse.nodes())
avg_clus_rand = float(m)/(n*(n-1)/2.)


#### powerlaw analysis
import powerlaw
data = in_hist # data can be list or numpy array
results = powerlaw.Fit(data, discrete=True)
print results.power_law.alpha
print results.power_law.xmin
print results.power_law.xmax
for alt in ['exponential', 'lognormal', 'stretched_exponential']:
    R, p = results.distribution_compare('truncated_power_law', alt)
    print alt, R, p


#test tweets per user function
twpusr = np.sum(TM_dict['TM_sparse']['tweets_matrix'], axis=1)
tw_values = sorted(set(twpusr)) 
tw_hist = [in_degrees.values().count(x) for x in in_values]

data = tw_hist # data can be list or numpy array
results = powerlaw.Fit(data, discrete=True)
print results.power_law.alpha
print results.power_law.xmin
print results.power_law.xmax
for alt in ['exponential', 'lognormal', 'stretched_exponential']:
    R, p = results.distribution_compare('truncated_power_law', alt)
    print alt, R, p



#TODO:
#LDA?





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