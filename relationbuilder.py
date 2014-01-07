# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:36:10 2013

@author: kurka
"""

import tweepy
import time
import json
from PersistencyLayer import TweetsPersister

json_fp = open('credentials.json')
cred = json.load(json_fp)

auth = tweepy.OAuthHandler(cred['twitter']['consumer_key'], cred['twitter']['consumer_secret'])
auth.set_access_token(cred['twitter']['access_token'], cred['twitter']['access_token_secret'])

api = tweepy.API(auth)

rl = api.rate_limit_status()
print(rl['resources']['friendships']['/friendships/show'])
if rl['resources']['friendships']['/friendships/show']['reset'] == 0:
    time_to_reset = rl['resources']['friendships']['/friendships/show']['reset']-int(time.time())+11
    print "dormindo por %s segundos esperando o reset" % time_to_reset
    if time_to_reset > 0:
        time.sleep(time_to_reset)
    else:
        time.sleep(10)

persister = TweetsPersister() #connect to database

while True:
    #try to get new user
    new_user = persister.loadUnprocessedUser2() 
        
    if new_user and new_user != 6017542:        
        print 'building network for user %s' % new_user
                
        
        try:
            #check if user is related to source
            relation = api.show_friendship(source_id=6017542, target_id=new_user)
            if relation[1].following == True:
                persister.saveFollower(new_user)
                
            #checa se ainda da pra fazer requisicoes
            rl = api.rate_limit_status()
            if rl['resources']['friendships']['/friendships/show']['remaining'] == 0:
                time_to_reset = rl['resources']['friendships']['/friendships/show']['reset']-int(time.time())+11
                print "dormindo por %s segundos esperando o reset" % time_to_reset
                if time_to_reset > 0:
                    time.sleep(time_to_reset)
                else:
                    time.sleep(10)
            else:
                time.sleep(5)
            
            rl = api.rate_limit_status()
            while rl['resources']['friendships']['/friendships/show']['remaining'] == 0:
                time.sleep(10)
                rl = api.rate_limit_status()

            #registra no banco de dados que usuario ja foi processado
            persister.saveProcessedUser(new_user, 4)
        except (tweepy.TweepError), e:
            print(e)
            if rl['resources']['friendships']['/friendships/show']['remaining'] > 0:
                #algum erro que nao o rate limit fez ele entrar aqui
                persister.saveProcessedUser(new_user, 5)
                time.sleep(60)
                continue
            elif rl['resources']['friendships']['/friendships/show']['remaining'] == 0:
                time_to_reset = rl['resources']['friendships']['/friendships/show']['reset']-int(time.time())+21
                print "dormindo por %s segundos esperando o reset" % time_to_reset
                if time_to_reset > 0:
                    time.sleep(time_to_reset)


    else:
        print ">>>Relation Builder desocupado!"
        time.sleep(60)
            
            
            
            
            
#            try:
#                rl = api.rate_limit_status()
#                print(rl['resources']['friendships']['/friendships/show'])
#                if rl['resources']['friendships']['/friendships/show']['remaining'] == 0:
#                    time_to_reset = rl['resources']['friendships']['/friendships/show']['reset']-int(time.time())+11
#                    print "dormindo por %s segundos esperando o reset" % time_to_reset
#                    if time_to_reset > 0:
#                        time.sleep(time_to_reset)
#                print(page)
#                follower = api.friendships_ids(id=new_user, cursor=page)
#                print(follower)
#                if follower:
#                    followers += follower
#                    time.sleep(5)
#                else:
#                    break
#                page += 1 #next page
#            except (tweepy.TweepError), e:
#                print(e)
#                if rl['resources']['friendships']['/friendships/show']['remaining'] > 0:
#                    #algum erro que nao o rate limit fez ele entrar aqui
#                    persister.saveProcessedUser(new_user, 2)
#                    time.sleep(60)
#                    break
#                else:#falta de recursos, tenta de novo
#                    continue
#        if followers:
#            #salva lista em arquivo
#            f = open("net/"+str(new_user), "w")
#            json.dump(followers, f)
#            f.close()
#            #registra no banco de dados que usuario ja foi processado
#            persister.saveProcessedUser(new_user, 1)

