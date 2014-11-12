# -*- coding: utf-8 -*-



import tweepy
import time
import json
import sys
from persistencylayer import TweetsPersister
from pymongo import MongoClient

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

        self.persister = TweetsPersister(credentials) #connect to mysql database

        json_fp = open(credentials, 'r')
        cred = json.load(json_fp)

        #connect to mongodb database        
        #self.relationsdb = MongoClient(cred['mongo']['host'], cred['mongo']['port']).twitter
        

        #get credentials, to connect with Twitter API
        self.apps = cred['twitter_apps']
          
        n_apps = len(self.apps)

        self.apps_resources = [0] * n_apps  #number of requisitions left
        self.apps_timeout = [0] * n_apps    #time where requisitions will be restored
        
        for app in range(n_apps):
            self.connect(app)               #connect once in all apps, to get right info

        self.current_app = 0                #app currently connected to API
        
          
        self.connect(self.current_app) #connect to first app of the list
        self.api_waiter()
        self.buildnetwork(self.mode, self.direction)
            
      
      
      


    def connect(self, app_id=0):
        """connect to Twitter API and to database"""        
    
        #OAuth autentication with Twitter API
        auth = tweepy.OAuthHandler(self.apps[app_id]['consumer_key'], self.apps[app_id]['consumer_secret'])
        auth.set_access_token(self.apps[app_id]['access_token'], self.apps[app_id]['access_token_secret'])
    
        self.api = tweepy.API(auth)
        self.current_app = app_id

    
        #show the resources available
        rl = self.api.rate_limit_status()
        info = rl['resources'][self.direction]['/'+self.direction+'/ids']
        print "current app: %d" %self.current_app        
        print "remaining: %d" %info['remaining']
        print "reset time: %s" %time.ctime(info['reset'])
        self.apps_resources[app_id] = info['remaining']
        self.apps_timeout[app_id] = info['reset']


        
    def api_waiter(self, wait_anyway=True):
        """Handles with api rating limits"""
    
        rl = self.api.rate_limit_status()
        info = rl['resources'][self.direction]['/'+self.direction+'/ids']
        remaining_resources = info['remaining']
        self.apps_resources[self.current_app] = remaining_resources
        self.apps_timeout[self.current_app] = info['reset']
        
        #check if there is still resources
        if remaining_resources > 0:
            if wait_anyway:
                time.sleep(3)
                return
            else:
                return #just try again
        
        elif remaining_resources == 0:
            #try to change the app 1
            for app_id, remaining in enumerate(self.apps_resources):
                if remaining > 0:
                    self.connect(app_id)
                    self.api_waiter()
                    return

            #try to change the app 2            
            #if there isn't any app with resources, choose the closer to reset and wait
            oldest_app = self.apps_timeout.index(min(self.apps_timeout))
            if oldest_app != self.current_app:
                self.connect(oldest_app)
                self.api_waiter()
                return
            
            elif oldest_app == self.current_app:
                while remaining_resources == 0:
                    time_to_reset = rl['resources'][self.direction]['/'+self.direction+'/ids']['reset']-int(time.time())+1 
                    print "dormindo por %s segundos esperando o reset" % time_to_reset
                    if time_to_reset > 0:
                        time.sleep(time_to_reset)
                    else:
                        time.sleep(5) #sleep at least 5 seconds
            
                    #update rate limits        
                    rl = self.api.rate_limit_status()
                    remaining_resources = rl['resources'][self.direction]['/'+self.direction+'/ids']['remaining']
                    self.apps_resources[self.current_app] = remaining_resources
                    self.apps_timeout[self.current_app] = rl['resources'][self.direction]['/'+self.direction+'/ids']['reset']
                         
        return
    
    
    
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

    
        while True:
            if mode=='active': #get new users from database              
                new_user = self.persister.loadUnprocessedUser(direction, origin=mode)
                print "new_user", new_user
            elif mode=='queue': #get new users from queue
                #try first to get unprocessed users from Users database
                new_user = self.persister.loadUnprocessedUser(direction, origin=mode)
                if not new_user:
                    #try to find suitable user in the queue
                    new_user = self.persister.popQueueUser() 
                    
                    #look for users not already in database
                    while new_user and self.persister.findUser(new_user):
                        new_user = self.persister.popQueueUser()
                    
                    if new_user:
                        #insert user in database
                        try:
                            userobject = self.api.get_user(new_user)
                            self.persister.insertUserObject(userobject, origin=2)
                        except (tweepy.TweepError), e:
                            print e
                            continue
                    
                                 
                    
                
            
            if new_user:                        
                print 'building network for user %s' % new_user
                
                #TODO: update user info
                
                #get user's connections
                connections = []
                try:
                    next_cursor=-1
                    while(next_cursor!=0):
                        #request = eval(request_cmd) #evaluate everytime, in case app has changed
                        ids, (_, next_cursor) = request(id=new_user, cursor=next_cursor)
                        
                        connections += ids
                        print len(connections), "connections"

                        #wait if needed to do more requests                        
                        self.api_waiter()
                        
                        #update request object, case api has changed
                        if direction == 'followers':
                            request = self.api.followers_ids                     
                        elif direction == 'friends':
                            request = self.api.friends_ids
                    

        
        
                    #save follower list in file
                    if direction == 'followers':
                        f = open("net/"+str(new_user), "w")
                    elif direction == 'friends':
                        f = open("netr/"+str(new_user), "w")
                    json.dump(connections, f)
                    f.close()
                    
                    
                    
                    #register in database that user was processed
                    self.persister.saveProcessedUser(new_user, direction)  

                    
#                    if self.persister.isUserProcessed(new_user): #check if everything was collected already (friends & followers)                
#                        f = open('net/'+str(new_user))
#                        followers = set(json.load(f))
#                        f.close()
#                        
#                        f = open('netr/'+str(new_user))
#                        friends = set(json.load(f))
#                        f.close()                        
#                        
#                        true_friends = list(followers & friends)
#                        
#                        self.persister.pushQueueUsers(true_friends)
                        
                    
                except (tweepy.TweepError), e:                   
                    print(e)
                  
                    if type(e.message[0]) == dict and e.message[0]['code'] == 88:
                        print "skipping user error limit"
                        time.sleep(60)
                    else:
                        self.persister.saveProcessedUser(new_user, error=True) 
                    self.api_waiter()
                    continue
                    #TODO: treat specially lack of resources error
        
            elif new_user == None:
                print ">>>Network Builder desocupado!"
                time.sleep(10)
                break  
        
        
def main(argv):
    
    #import cProfile
    #cProfile.run("program(sys.argv)", "test.profile")
    if len(argv) != 4:
        print "Usage: python networkbuilder.py [credentials_file] [active|queue] [followers|friends]"
        return
    
    NetworkBuilder(argv[1], mode=argv[2], direction=argv[3])    

if __name__ == "__main__":
    main(sys.argv)
