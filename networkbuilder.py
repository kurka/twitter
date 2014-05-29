# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:36:10 2013

@author: kurka
"""

import tweepy
import time
import json
from persistencylayer import TweetsPersister

FOLHA = '14594813'
ESTADAO = '9317502'
UOLNOT = '14594698'
G1 = '8802752'
R7 = '65473559'
SOURCE_IDS = [int(FOLHA), int(ESTADAO), int(UOLNOT), int(G1), int(R7)]



def api_waiter(wait_anyway=True):
    """Handles with api rating limits"""

    rl = api.rate_limit_status()
    remaining_resources = rl['resources']['followers']['/followers/ids']['remaining']
    
    #check if there is still resources
    if remaining_resources > 0:
        if wait_anyway:
            time.sleep(60) #wait for 60 seconds, before using it again
        else:
            return #just try again
    
    else:
        while remaining_resources == 0:
            time_to_reset = rl['resources']['followers']['/followers/ids']['reset']-int(time.time())+2 
            print "dormindo por %s segundos esperando o reset" % time_to_reset
            if time_to_reset > 0:
                time.sleep(time_to_reset)
            else:
                time.sleep(5) #sleep at least 5 seconds
    
            #update rate limits        
            rl = api.rate_limit_status()
            remaining_resources = rl['resources']['followers']['/followers/ids']['remaining']



#get credentials, to connect with Twitter API
json_fp = open('credentials.json')
cred = json.load(json_fp)

#OAuth autentication with Twitter API
auth = tweepy.OAuthHandler(cred['twitter']['consumer_key'], cred['twitter']['consumer_secret'])
auth.set_access_token(cred['twitter']['access_token'], cred['twitter']['access_token_secret'])

api = tweepy.API(auth)

#show the resources available
rl = api.rate_limit_status()
print(rl['resources']['followers']['/followers/ids'])
api_waiter(False)
persister = TweetsPersister() #connect to database
    

while True:
    #try to get new user
    new_user = persister.loadUnprocessedRecurrentUser() #try to get recurrent users first
    if not new_user:
        new_user = persister.loadUnprocessedUser()
        
    if new_user and new_user not in SOURCE_IDS:
        print 'building network for user %s' % new_user
                
        #1 check if user is related to source
#            relation = api.show_friendship(source_id=SOURCE_ID, target_id=new_user)
#            if relation[1].following == True:
#                persister.saveFollower(new_user)
        
        #2 get user's followers
        followers = []
        try:
            for follower in tweepy.Cursor(api.followers_ids, id=new_user).pages():
                followers += follower
                print len(followers), "followers"
                
                #wait to do more requests
                api_waiter()

            #salva lista em arquivo
            f = open("net/"+str(new_user), "w")
            json.dump(followers, f)
            f.close()
            #registra no banco de dados que usuario ja foi processado
            persister.saveProcessedUser(new_user)
        except (tweepy.TweepError), e:
            print(e)
            persister.saveProcessedUser(new_user, 7) #TODO: TEMP! 
            api_waiter()
            continue
            #TODO: treat specially lack of resources error

    elif new_user in SOURCE_IDS:
        persister.saveProcessedUser(new_user, 3)
    else:
        print ">>>Network Builder desocupado!"
        time.sleep(10)

            
            
            
            


