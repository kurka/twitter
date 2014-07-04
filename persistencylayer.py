import MySQLdb
import json
from warnings import filterwarnings
import random


#TODO: consertar identacao!!

class TweetsPersister():
   def __init__(self, cred='credentials.json'):
      json_fp = open(cred)
      self.cred = json.load(json_fp)
      self.connect()
      filterwarnings('ignore', category = MySQLdb.Warning)
    
   def connect(self):
      self.db = MySQLdb.connect(host = self.cred['db']['host'],
                                db = self.cred['db']['db'],
                                user = self.cred['db']['user'],
                                passwd = self.cred['db']['password'],
                                charset = self.cred['db']['charset'])

   def query(self, sql, params):
      try:
        cursor = self.db.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
      except (AttributeError, MySQLdb.OperationalError):
        self.connect()
        cursor = self.db.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
      return cursor
      
   def commit(self):
      self.db.commit()

   def insertRawTweet(self, string):
      self.query("INSERT INTO tweets_raw VALUES (%s)", string)
      self.commit()
      return

   def insertParsedTweet(self, tweet):
     """
     Create/update tweet on database.
     @param Populated tweet.
     """


     tweet['user_id'] = tweet['user']['id']

     # create tweet
     try:
        c = self.db.cursor()
     except (AttributeError, MySQLdb.OperationalError):
        self.connect()
        c = self.db.cursor()
      
     data = (
        tweet['id'],
        tweet['text'].encode('utf-8'),
        tweet['source'],
        (1 if tweet['truncated'] else 0),
        tweet['in_reply_to_status_id'],
        tweet['in_reply_to_user_id'],
        tweet['in_reply_to_screen_name'],
        tweet['user_id'],
        tweet['retweet_count'],
        tweet['favorite_count'],
        (1 if tweet['favorited'] else 0),
        (1 if tweet['retweeted'] else 0),
        (1 if ('possibly_sensitive' in tweet and tweet['possibly_sensitive']) else 0),
        tweet['lang'],
        (tweet['retweeted_status']['id'] if 'retweeted_status' in tweet else 0),
        tweet['created_at']
     )
     c.execute("INSERT INTO tweets (tweet_id, tweet_text, tweet_source, tweet_truncated, tweet_in_reply_to_status_id, tweet_in_reply_to_user_id, tweet_in_reply_to_screen_name, user_id, tweet_retweet_count, tweet_favorite_count, tweet_favorited, tweet_retweeted, tweet_possibly_sensitive, tweet_lang, tweet_retweeted_status_id, tweet_created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", data)

     if 'entities' in tweet:
        if 'urls' in tweet['entities']:
           for url in tweet['entities']['urls']:
              url_data = (
                 tweet['id'],
                 url['url'],
                 url['display_url'],
                 url['expanded_url'],
                 '%d;%d' % (url['indices'][0], url['indices'][1])
              )
              c.execute("INSERT INTO tweets_url (tweet_id, tweet_url_url, tweet_url_display_url, tweet_url_expanded_url, tweet_url_indices) VALUES (%s, %s, %s, %s, %s)", url_data)

        if 'user_mentions' in tweet['entities']:
           for user_mention in tweet['entities']['user_mentions']:
              user_mention_data = (
                 tweet['id'],
                 user_mention['id'],
                 user_mention['screen_name'],
                 user_mention['name'],
                 '%d;%d' % (user_mention['indices'][0], user_mention['indices'][1])
              )
              c.execute("INSERT INTO tweets_usermention (tweet_id, user_id, user_screen_name, user_name, tweet_usermention_indices) VALUES (%s, %s, %s, %s, %s)", user_mention_data)

        if 'hashtags' in tweet['entities']:
           for hashtag in tweet['entities']['hashtags']:
              hashtag_data = (
                 tweet['id'],
                 hashtag['text'],
                 '%d;%d' % (hashtag['indices'][0], hashtag['indices'][1])
              )
              c.execute("INSERT INTO tweets_hashtag (tweet_id, hashtag_text, hashtag_indices) VALUES (%s, %s, %s)", hashtag_data)

     self.commit()
     return

   def insertUser(self, user, origin=1):
      data = (
         user['id'],
         user['name'],
         user['screen_name'],
         user['location'],
         user['description'],
         user['url'],
         user['followers_count'],
         user['friends_count'],
         user['listed_count'],
         user['created_at'],
         user['favourites_count'],
         user['utc_offset'],
         user['time_zone'],
         (1 if user['geo_enabled'] else 0),
         (1 if user['verified'] else 0),
         user['statuses_count'],
         user['lang'],
         (1 if user['contributors_enabled'] else 0),
         0, #got_followers
         0, #got_following
         origin
      )
      self.query("INSERT INTO users (user_id, user_name, user_screen_name, user_location, user_description, user_url, user_followers_count, user_friends_count, user_listed_count, user_created_at, user_favourites_count, user_utc_offset, user_time_zone, user_geo_enabled, user_verified, user_statuses_count, user_lang, user_contributors_enabled, got_followers, got_friends, origin, inserted_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())", data)
      self.commit()
      return
      
   def insertUserObject(self, user, origin=1):
      """insert into database tweepy 'user' object, returned by get_user function """
      data = (
         user.id,
         user.name,
         user.screen_name,
         user.location,
         user.description,
         user.url,
         user.followers_count,
         user.friends_count,
         user.listed_count,
         user.created_at.strftime('%a %b %d %H:%M:%S +0000 %Y'),
         user.favourites_count,
         user.utc_offset,
         user.time_zone,
         (1 if user.geo_enabled else 0),
         (1 if user.verified else 0),
         user.statuses_count,
         user.lang,
         (1 if user.contributors_enabled else 0),
         0, #got_followers
         0, #got_following
         origin
      )
      self.query("INSERT INTO users (user_id, user_name, user_screen_name, user_location, user_description, user_url, user_followers_count, user_friends_count, user_listed_count, user_created_at, user_favourites_count, user_utc_offset, user_time_zone, user_geo_enabled, user_verified, user_statuses_count, user_lang, user_contributors_enabled, got_followers, got_friends, origin, inserted_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())", data)
      self.commit()
      return


   def loadTweet(self, tweet_id):
      """
      Tweet loader.
      @param tweet_id
      @return Populated tweet dictionary, if tweet found. "None" otherwise.
      """
      sql = "SELECT tweet_text, tweet_source, user_id, tweet_retweeted_status_id FROM tweet WHERE tweet_id = %s"
      data = (tweet_id,)
      c = self.query(sql, data)
      
      row = c.fetchone()
      if row is None:
         return None

      tweet = {
         'id': tweet_id,
         'text': row[0],
         'source': row[1],
         'user_id': row[2],
         'retweeted_status_id': row[3],
      }

      return tweet
      
   def loadTweetsOfUser(self, user):
      """
      Tweet loader.
      @param tweet_id
      @return Populated tweet dictionary, if tweet found. "None" otherwise.
      """
      sql = "SELECT tweet_id, tweet_text, tweet_created_at FROM tweets WHERE user_id = %s order by inserted_at"
      data = (user,)
      c = self.query(sql, data)
      
      rows = c.fetchall()
      if rows is None:
         return None

      tweets = []
      for row in rows:
          tweet = {
              'tweet_id': row[0],
              'text': row[1],
              'time': row[2],
              'user_id': user
             }
          tweets.append(tweet)

      return tweets

   def loadRetweets(self, tweet_id):
      """
      Tweet loader.
      @param tweet_id
      @return Populated tweet dictionary, if tweet found. "None" otherwise.
      """
      sql = "SELECT tweet_id, tweet_text, tweet_created_at, user_id FROM tweets WHERE tweet_retweeted_status_id = %s"
      data = (tweet_id,)
      c = self.query(sql, data)
      
      rows = c.fetchall()
      if rows is None:
         return None

      retweets = []
      for row in rows:
          tweet = {
              'tweet_id': row[0],
              'text': row[1],
              'time': row[2],
              'user_id': row[3]
             }
          retweets.append(tweet)

      return retweets


   def loadUsers(self):
      """
      User loader
      @return user_id's list, if found. "None" otherwise.
      """
      sql = "SELECT user_id FROM users"
      data = ()
      c = self.query(sql, data)
      rows = c.fetchall()
      #flattening result
      users = [element for tupl in rows for element in tupl]

      return users
     

      
   def loadUnprocessedUser(self, direction, origin='active'):
      """
      Load unprocessed user
      @return Single user_id
      """
      
      if origin=='active':
          originid=1
      elif origin=='queue':
          originid=2
      
      
      if direction == 'followers':
          sql = "SELECT user_id FROM users WHERE origin=%s AND got_followers=0 ORDER BY recurrent DESC, inserted_at LIMIT 1"
      elif direction == 'friends':
          sql = "SELECT user_id FROM users WHERE origin=%s AND got_friends=0 ORDER BY recurrent DESC, inserted_at LIMIT 1"
      data = (originid,)
      c = self.query(sql, data)
      row = c.fetchone()
      if row == None:
          return None
      else:
          return row[0]
          
   def popQueueUser(self):
       """
       Load user from table users_queue.
       Delete user, after loading it (queue operation)
       """
       #TODO: deal with concurrency 
       #(maybe with locks: http://dev.mysql.com/doc/refman/5.1/en/miscellaneous-functions.html#function_get-lock
       #http://stackoverflow.com/questions/423111/whats-the-best-way-of-implementing-a-messaging-queue-table-in-mysql)        
        
        
       #get the top of the queue
       sql = "SELECT user_id FROM users_queue where hits > 100"
       data = ()
       c = self.query(sql, data)
       try:
           row = random.choice(c.fetchall())
       except:
           sql = "SELECT user_id FROM users_queue order by hits desc limit 1"
           c = self.query(sql, data)
           row = c.fetchone()
           
       if row == None: #queue is empty
           return None
       else: #remove from queue
           user = row[0]
           sql = "DELETE FROM users_queue WHERE user_id=%s"
           data = (user,)
           c = self.query(sql, data)
           self.commit()
           return user


#   def pushQueueUser(self, users):
#       """
#       Push list of users into the queue.
#       
#       Args:
#           users: list of ids to be pushed
#       """
#       
#       for chunk in xrange(0, len(users), 1000): #divide list in chunks of 1000
#           #build string with numbers in brackets
#           strusers = "(" + "),(".join(map(str, users[chunk:chunk+1000])) + ")" 
#           sql = "INSERT IGNORE INTO users_queue(user_id) VALUES %s;" %strusers
#           data = ()
#           self.query(sql, data)
#           self.commit()
 
   def pushQueueUsers(self, users):
       """
       Push list of users into the queue.
       
       If user already exists, increment "hit" field.
       Args:
           users: list of ids to be pushed
       """
       chunksize = 100
       for chunk in xrange(0, len(users), chunksize): #divide list in chunks
           #build string with numbers in brackets
           users_set = users[chunk:chunk+chunksize]
           strusers = str(map(None, users_set, [1]*len(users_set))).strip('[]')
           sql = "INSERT INTO users_queue (user_id, hits) VALUES %s ON DUPLICATE KEY UPDATE hits = hits + 1" %strusers
           data = ()
           self.query(sql, data)
           self.commit()       
       
       return 
       
       
          
   def saveProcessedUser(self, user_id, direction='followers', error=False):
       
      if not error:
          if direction == 'followers':
              sql = "UPDATE users SET got_followers=1 WHERE user_id=%s"
          elif direction == 'friends':
                  sql = "UPDATE users SET got_friends=1 WHERE user_id=%s"
      else:
          sql = "UPDATE users SET got_followers=2, got_friends=2 WHERE user_id=%s"
          
      data = (user_id,)
      self.query(sql, data)
      self.commit()
      return

   def saveRecurrentUser(self, user_id, value=1):
      sql = "UPDATE users SET recurrent=%s WHERE user_id=%s"
      data = (value, user_id)
      self.query(sql, data)
      self.commit()
      return 

   def saveFollower(self, user_id):   
      sql = "UPDATE users SET friend_of_source=1 WHERE user_id=%s"
      data = (user_id,)
      self.query(sql, data)
      self.commit()
      return 

   def isUserProcessed(self, user_id):
       """
       Check if both fields 'got_followers' and 'got_friends' are equal 1,
       meaning that we already got information from followers and friends of user
       """
       sql = "SELECT got_followers, got_friends FROM users WHERE user_id=%s"
       data = (user_id,)
       c = self.query(sql, data)
       row = c.fetchone()
       
       if row[0]==1 and row[1]==1:
           return True
       else:
           return False
       


   def findUser(self, user_id):
      """
      User loader.
      @param user_id
      @return user's time of insertion, if found. "None" otherwise.
      """
      
      sql = "SELECT inserted_at FROM users WHERE user_id = %s"
      params = (user_id,)
      c = self.query(sql, params)
      row = c.fetchone()
      
      if row is None:
         return None
      else:
          return row[0]

      
      
   def loadUser(self, user_id):
      """
      User loader.
      @param user_id
      @return Populated user dictionary, if found. "None" otherwise.
      """
      sql = "SELECT user_name, user_screen_name, user_location, user_description, user_url, user_followers_count, user_friends_count, user_favourites_count FROM users WHERE user_id = %s"
      params = (user_id,)
      c = self.query(sql, params)
      row = c.fetchone()
      
      if row is None:
         return None

      user = {
         'id': user_id,
         'name': row[0],
         'screen_name': row[1],
         'location': row[2],
         'description': row[3],
         'url': row[4],
         'followers_count': row[5],
         'friends_count': row[6],
         'favorited_count': row[7]
         #'processed': row[-2],
         #'friend_of_source' : row[-1]
      }

      return user



   def updateTweetCreatedAt(self, tweet):
      sql = "UPDATE tweet SET tweet_created_at = %s WHERE tweet_id = %s"
      data = (
         tweet['created_at'],
         tweet['id']
      )
      self.query(sql, data)
      self.commit()
      return


#   #def createTables(self):
#   #\colocar aqui
#
#    CREATE TABLE `tweets` (
#      `tweet_id` bigint(20) unsigned NOT NULL,
#      `tweet_text` varchar(255) DEFAULT NULL,
#      `tweet_source` varchar(256) DEFAULT NULL,
#      `tweet_truncated` smallint(6) DEFAULT NULL,
#      `tweet_in_reply_to_status_id` bigint(20) unsigned DEFAULT NULL,
#      `tweet_in_reply_to_user_id` bigint(20) unsigned DEFAULT NULL,
#      `tweet_in_reply_to_screen_name` varchar(128) DEFAULT NULL,
#      `user_id` bigint(20) DEFAULT NULL,
#      `tweet_retweet_count` int(10) unsigned DEFAULT NULL,
#      `tweet_favorite_count` int(10) unsigned DEFAULT NULL,
#      `tweet_favorited` smallint(6) DEFAULT NULL,
#      `tweet_retweeted` smallint(6) DEFAULT NULL,
#      `tweet_possibly_sensitive` smallint(6) DEFAULT NULL,
#      `tweet_lang` varchar(8) DEFAULT NULL,
#      `tweet_retweeted_status_id` bigint(20) unsigned DEFAULT NULL,
#      `tweet_created_at` varchar(128) DEFAULT NULL
#    );
#
#   CREATE TABLE `tweets_url` (
#      `tweet_url_id` bigint(20) unsigned NOT NULL DEFAULT '0',
#      `tweet_id` bigint(20) unsigned DEFAULT NULL,
#      `tweet_url_url` varchar(128) CHARACTER SET utf8 DEFAULT NULL,
#      `tweet_url_display_url` varchar(128) CHARACTER SET utf8 DEFAULT NULL,
#      `tweet_url_expanded_url` varchar(128) CHARACTER SET utf8 DEFAULT NULL,
#      `tweet_url_indices` varchar(45) CHARACTER SET utf8 DEFAULT NULL
#    );
#
#  CREATE TABLE `tweets_usermention` (
#      `tweet_usermention_id` bigint(20) unsigned NOT NULL DEFAULT '0',
#      `tweet_id` bigint(20) unsigned DEFAULT NULL,
#      `user_id` bigint(20) unsigned DEFAULT NULL,
#      `user_screen_name` varchar(256) CHARACTER SET utf8 DEFAULT NULL,
#      `user_name` varchar(256) CHARACTER SET utf8 DEFAULT NULL,
#      `tweet_usermention_indices` varchar(45) CHARACTER SET utf8 DEFAULT NULL
#    );
#    
#    
#    CREATE TABLE `tweets_hashtag` (
#      `tweet_hashtag_id` bigint(20) unsigned NOT NULL DEFAULT '0',
#      `tweet_id` bigint(20) unsigned DEFAULT NULL,
#      `hashtag_text` varchar(140) CHARACTER SET utf8 DEFAULT NULL,
#      `hashtag_indices` varchar(128) CHARACTER SET utf8 DEFAULT NULL
#    );
#
#    CREATE TABLE `users` (
#      `user_id` bigint(20) unsigned NOT NULL,
#      `user_name` varchar(256) CHARACTER SET utf8 NOT NULL,
#      `user_screen_name` varchar(256) CHARACTER SET utf8 NOT NULL,
#      `user_location` varchar(256) CHARACTER SET utf8 DEFAULT NULL,
#      `user_description` varchar(1024) CHARACTER SET utf8 DEFAULT NULL,
#      `user_url` varchar(256) CHARACTER SET utf8 DEFAULT NULL,
#      `user_followers_count` int(10) unsigned DEFAULT NULL,
#      `user_friends_count` int(10) unsigned DEFAULT NULL,
#      `user_listed_count` int(10) unsigned DEFAULT NULL,
#      `user_created_at` varchar(128) CHARACTER SET utf8 DEFAULT NULL,
#      `user_favourites_count` int(10) unsigned DEFAULT NULL,
#      `user_utc_offset` int(11) DEFAULT NULL,
#      `user_time_zone` varchar(128) CHARACTER SET utf8 DEFAULT NULL,
#      `user_geo_enabled` smallint(6) DEFAULT NULL,
#      `user_verified` smallint(6) DEFAULT NULL,
#      `user_statuses_count` int(10) unsigned DEFAULT NULL,
#      `user_lang` varchar(10) CHARACTER SET utf8 DEFAULT NULL,
#      `user_contributors_enabled` smallint(6) DEFAULT NULL,
#      `user_processed` smallint(6) DEFAULT 0,
#        `got_followers` smallint(6) DEFAULT 0,
#        `got_friends` smallint(6) DEFAULT 0,
#        `origin` smallint(6) DEFAULT 0, #1 active, 2 queue 
#      `friend_of_source` smallint(6) DEFAULT 0,
#      `recurrent` smallint(6) DEFAULT 0
#    );
#    (FALTA TABLE tweets_raw)

#CREATE TABLE 'graph_edges' (
#        'from_node' bigint(20) unsigned NOT NULL,
#        'to_node' bigint(20) unsigned NOT NULL
#        );


#create table users_queue(
#  id int(20) auto_increment not null primary key,
#  user_id bigint(20) unsigned not null,
#  hits int(10) not null default 0,  
#  unique key user_id(user_id)
#  
#);


#SELECT STR_TO_DATE('Sun May 18 11:59:09 +0000 2014', '%a %b %d %H:%i:%s +0000 %Y');
#update users u set inserted_at = (select min(STR_TO_DATE(tweet_created_at, '%a %b %d %H:%i:%s +0000 %Y')) from tweets where tweets.user_id = u.user_id);



