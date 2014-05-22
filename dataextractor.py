# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:56:09 2013

@author: kurka
"""
import json
from PersistencyLayer import TweetsPersister
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

FOLHA = '14594813'
ESTADAO = '9317502'
UOLNOT = '14594698'
G1 = '8802752'
R7 = '65473559'

json_fp = open('credentials.json')
cred = json.load(json_fp)

persister = TweetsPersister()


class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    def on_data(self, data):
        parsed = json.loads(data)    
        
        user_id = int(parsed['user']['id_str'])
        if user_id in userslist:
            print ">>>>>USUARIO REPETIDO:",  parsed['user']['name'], parsed['user']['screen_name']

        else:
            print "USUARIO NOVO:",  parsed['user']['name'], parsed['user']['screen_name']        
        
        
        #guardar tweet no banco de dados geral
        #TODO: fazer isso como processo independente!
        persister.insertRawTweet(data)
        persister.insertParsedTweet(parsed)
       
        #ver se o id eh unico e incrementar ids        
        if user_id not in userslist:
            persister.insertUser(parsed['user'])
            userslist.add(user_id)
            
        else:
            persister.saveRecurrentUser(user_id)
            
            
        print parsed['text']
        print

        return True

    def on_error(self, status):
        print "error: %d" % status

if __name__ == '__main__':


    userslist = persister.loadUsers()
    userslist = set(userslist) #use set, to make it much more efficient
    print("lista carregada!")

    l = StdOutListener()
    auth = OAuthHandler(cred['twitter']['consumer_key'], cred['twitter']['consumer_secret'])
    auth.set_access_token(cred['twitter']['access_token'], cred['twitter']['access_token_secret'])

    stream = Stream(auth, l)
    stream.filter(follow=[FOLHA, ESTADAO, UOLNOT, G1, R7])
    
    
