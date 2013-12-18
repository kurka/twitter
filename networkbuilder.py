# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:36:10 2013

@author: kurka
"""

import tweepy
import time
import json
from PersistencyLayer import TweetsPersister

consumer_key="Iy0FIPNUrVHmznrTRJOlg"
consumer_secret="5J489FFczEl4JxSg1CvprUeXy4XHcrz8jXcYpZn9Gqs"

access_token="17354174-w4rtR2vAA5XGPviooDmoK6bBTXhwnMR1GiycA1YlQ"
access_token_secret="oOgag8mNHJdCkUZPYl4EzjLk4qyYWhGKYTsMQZ0SDrM"
json_fp = open('credentials.json')
cred = json.load(json_fp)

auth = tweepy.OAuthHandler(cred['twitter']['consumer_key'], cred['twitter']['consumer_secret'])
auth.set_access_token(cred['twitter']['access_token'], cred['twitter']['access_token_secret'])

api = tweepy.API(auth)

rl = api.rate_limit_status()
print(rl['resources']['followers']['/followers/ids'])

persister = TweetsPersister() #connect to database

while True:
    #try to get new user

    new_user = 813286#persister.loadUnprocessedUser()
    if new_user:
        print 'building network for user %s' % new_user
        followers = []
        page = 1
        while True:
            try:
                for follower in tweepy.Cursor(api.followers_ids, id=new_user).pages():
                    rl = api.rate_limit_status()
                    print(rl['resources']['followers']['/followers/ids'])
                    followers += follower
                    print(len(followers))
                    time.sleep(5)
                #salva lista em arquivo
                f = open("net/"+str(new_user), "w")
                json.dump(followers, f)
                f.close()
                #registra no banco de dados que usuario ja foi processado
                persister.saveProcessedUser(new_user)
                break
            except (tweepy.TweepError), e:
                print(e)
                if rl['resources']['followers']['/followers/ids']['remaining'] > 0:
                    #algum erro que nao o rate limit fez ele entrar aqui
                    persister.saveProcessedUser(new_user, 2)
                    time.sleep(60)
                    break
                else:#falta de recursos, tenta de novo
                    time_to_reset = rl['resources']['followers']['/followers/ids']['reset']-int(time.time())+11
                    print "dormindo por %s segundos esperando o reset" % time_to_reset
                    if time_to_reset > 0:
                        time.sleep(time_to_reset)
                    continue
    
    else:
        print ">>>Network Builder desocupado!"
        time.sleep(10)
            
            
            
            
            
#            try:
#                rl = api.rate_limit_status()
#                print(rl['resources']['followers']['/followers/ids'])
#                if rl['resources']['followers']['/followers/ids']['remaining'] == 0:
#                    time_to_reset = rl['resources']['followers']['/followers/ids']['reset']-int(time.time())+11
#                    print "dormindo por %s segundos esperando o reset" % time_to_reset
#                    if time_to_reset > 0:
#                        time.sleep(time_to_reset)
#                print(page)
#                follower = api.followers_ids(id=new_user, cursor=page)
#                print(follower)
#                if follower:
#                    followers += follower
#                    time.sleep(5)
#                else:
#                    break
#                page += 1 #next page
#            except (tweepy.TweepError), e:
#                print(e)
#                if rl['resources']['followers']['/followers/ids']['remaining'] > 0:
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

