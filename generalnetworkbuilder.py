# -*- coding: utf-8 -*-




###1-pega novo usuario
#pega topo da fila
#confere se ja nao foi visitado



#pega lista de seguidores

#adiciona na fila
#guarda arquivo



import tweepy
import time
import json
import sys
from persistencylayer import TweetsPersister

FOLHA = '14594813'
ESTADAO = '9317502'
UOLNOT = '14594698'
G1 = '8802752'
R7 = '65473559'
SOURCE_IDS = [int(FOLHA), int(ESTADAO), int(UOLNOT), int(G1), int(R7)]

class NetworkBuilder():
    def __init__(self, credentials, mode, direction):
      FOLHA = '14594813'
      ESTADAO = '9317502'
      UOLNOT = '14594698'
      G1 = '8802752'
      R7 = '65473559'
      self.source_ids = [int(FOLHA), int(ESTADAO), int(UOLNOT), int(G1), int(R7)]
      self.direction = direction
      self.mode = mode
      
      self.connect(credentials)
      self.buildnetwork(self.mode, self.direction)
      
      


    def connect(self, credentials):
        """connect to Twitter API and to database"""
        
        #get credentials, to connect with Twitter API
        json_fp = open(credentials)
        cred = json.load(json_fp)
    
        #OAuth autentication with Twitter API
        auth = tweepy.OAuthHandler(cred['twitter']['consumer_key'], cred['twitter']['consumer_secret'])
        auth.set_access_token(cred['twitter']['access_token'], cred['twitter']['access_token_secret'])
    
        self.api = tweepy.API(auth)
    
        #show the resources available
        rl = self.api.rate_limit_status()
        print(rl['resources'][self.direction]['/'+self.direction+'/ids'])
        self.api_waiter(False)
        self.persister = TweetsPersister() #connect to database
    
        
    def api_waiter(self, wait_anyway=True):
        """Handles with api rating limits"""
    
        rl = self.api.rate_limit_status()
        remaining_resources = rl['resources'][self.direction]['/'+self.direction+'/ids']['remaining']
        
        #check if there is still resources
        if remaining_resources > 0:
            if wait_anyway:
                time.sleep(60) #wait for 60 seconds, before using it again
            else:
                return #just try again
        
        else:
            while remaining_resources == 0:
                time_to_reset = rl['resources']['followers']['/followers/ids']['reset']-int(time.time())+1 
                print "dormindo por %s segundos esperando o reset" % time_to_reset
                if time_to_reset > 0:
                    time.sleep(time_to_reset)
                else:
                    time.sleep(5) #sleep at least 10 seconds
        
                #update rate limits        
                rl = self.api.rate_limit_status()
                remaining_resources = rl['resources'][self.direction]['/'+self.direction+'/ids']['remaining']
    
    
    
    def buildnetwork(self, mode, direction): #active or queue
        """Get users connections in order to build a graph of relationships
        
        Args:
            mode: 'active' - build connections of users that twitted in the past, and already are in database
                  'queue' - do a breadth-first search in the network, starting from the root users
            direction: 'followers' - get users followers (out edges)
                       'friends' - get users friends (in edges)
    
        """
    
        if direction == 'followers':
            request = self.api.followers_ids                
            
        elif direction == 'friends':
            request = self.api.friends_ids
            
        #set the process id of each activity, to be registered in the database
        if mode == 'active' and direction == 'followers':
            start_pid = 0; final_pid = 1
        if mode == 'active' and direction == 'friends':
            start_pid = 1; final_pid = 4
        if mode == 'queue' and direction == 'followers':
            start_pid = 5; final_pid = 6
        if mode == 'queue' and direction == 'friends':
            start_pid = 6; final_pid = 7
        error_pid = 2
    
    
        while True:
            if mode=='active': #get new users from database
                new_user = self.persister.loadUnprocessedUser(user_processed=start_pid, recurrent=1) #try to get recurrent users first
                if not new_user:
                    new_user = self.persister.loadUnprocessedUser(user_processed=start_pid)
            
            elif mode=='queue': #get new users from queue
                if direction == 'followers':             
                    new_user = self.persister.popQueueUser() 
                    
                    #look for users not already in database
                    if new_user not in self.source_ids:  #temporary exception
                        while new_user and self.persister.findUser(new_user):
                            if new_user in self.source_ids:
                                break 
                            new_user = self.persister.popQueueUser()
                    
                        if new_user:
                            #insert user in database
                            try:
                                userobject = self.api.get_user(new_user)
                                self.persister.insertUserObject(userobject, start_pid)
                            except (tweepy.TweepError), e:
                                print e
                                continue
                elif direction == 'friends':
                    new_user = self.persister.loadUnprocessed(user_processed=start_pid)
                    
                                 
                            
                
            
            if new_user:                        
                print 'building network for user %s' % new_user
                
                #get user's connections
                connections = []
                try:
                    for userlist in tweepy.Cursor(request, id=new_user).pages():
                        connections += userlist
                        print len(connections), "connections"
                        
                        #put users in queue
                        if mode == 'queue':
                            self.persister.pushQueueUser(userlist)                        
                        
                        #wait to do more requests
                        self.api_waiter()
        
        
                    #save follower list in file
                    if direction == 'followers':
                        f = open("net/"+str(new_user), "w")
                    elif direction == 'friends':
                        f = open("netr/"+str(new_user), "w")
                        
                    json.dump(connections, f)
                    f.close()
                    
                    
                    
                    #register in database that user was processed
                    self.persister.saveProcessedUser(new_user, final_pid)                    
                    
                except (tweepy.TweepError), e:
                    print(e)
                    self.persister.saveProcessedUser(new_user, error_pid) 
                    self.api_waiter()
                    continue
                    #TODO: treat specially lack of resources error
        
            else:
                print ">>>Network Builder desocupado!"
                time.sleep(10)
        
        
def main(argv):
    
    #import cProfile
    #cProfile.run("program(sys.argv)", "test.profile")
    if len(argv) != 4:
        print "Usage: python networkbuilder.py [credentials_file] [active|queue] [followers|friends]"
        return
    
    nb = NetworkBuilder(argv[1], mode=argv[2], direction=argv[3])    
    #nb.buildnetwork()

if __name__ == "__main__":
    main(sys.argv)