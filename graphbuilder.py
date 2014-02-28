# -*- coding: utf-8 -*-

#import networkx as nx
#import os
import json
import pickle
from PersistencyLayer import TweetsPersister

json_fp = open('credentials.json')
cred = json.load(json_fp)

persister = TweetsPersister()


#0 - seleciona grupo de tweets
root_tweets = persister.loadTweetsOfUser(6017542)

tweets_collection = []
for root_tweet in root_tweets:
    retweets = persister.loadRetweets(root_tweet['tweet_id'])
    if retweets:
        retweets.insert(0, root_tweet) #insert root tweet at the beggining of the list
        tweets_collection.append(retweets)
    
users = list(set([item['user_id'] for sublist in tweets_collection for item in sublist]))

f1 = open('tweetcol.data', 'w')
f2 = open('users.data', 'w')

pickle.dump(tweets_collection, f1)
pickle.dump(users, f2)

f1.close()
f2.close()
    


#GClean = nx.DiGraph()
#GDirty = nx.DiGraph()
#
##1 - seleciona conjunto de nodes
#nodes_list = os.listdir('net')
#nodes_list = map(int, nodes_list)
#
#
#clean_edges = []
#dirty_edges = []
#
#total = float(len(nodes_list))
#count = 1
#for n in nodes_list:
#   print(str(count/total)+'%')
#   f = open('net/'+str(n))
#   neighbours = json.load(f)
#   for nei in neighbours:
#      if nei in nodes_list: #just add if node is active
#         clean_edges.append((n,nei))
#      dirty_edges.append((n, nei))
#   f.close()
#   count += 1
#
#GClean.add_edges_from(clean_edges)
#GDirty.add_edges_from(dirty_edges)

