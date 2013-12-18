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

json_fp = open('credentials.json')
cred = json.load(json_fp)

persister = TweetsPersister()

#TODO: git

class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    def on_data(self, data):
        parsed = json.loads(data)    
        
        user_id = int(parsed['user']['id_str'])
        print user_id

        
        #guardar tweet no banco de dados geral
        #TODO: fazer isso como processo independente!
        persister.insertRawTweet(data)
        persister.insertParsedTweet(parsed)
        
        #ver se o id eh unico e incrementar ids        
        if user_id not in userslist:
            persister.insertUser(parsed['user'])
            userslist.append(parsed['user']['id_str'])
            
        if user_id in userslist:
            print "USUARIO REPETIDO!"
            
            
        print parsed['text'] #TODO tratar caracteres mto bizarros

        return True

    def on_error(self, status):
        print "error: %d" % status

if __name__ == '__main__':


    userslist = persister.loadUsers()

    l = StdOutListener()
    auth = OAuthHandler(cred['twitter']['consumer_key'], cred['twitter']['consumer_secret'])
    auth.set_access_token(cred['twitter']['access_token'], cred['twitter']['access_token_secret'])

    stream = Stream(auth, l)
    stream.filter(follow=['783214'])
    #TODO: explorar novas formas de pegar conteudo
    #stream.filter(follow=[followedUserIds[0:2])])    
    
    
