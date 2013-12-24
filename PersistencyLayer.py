import MySQLdb
import json

class TweetsPersister():
   def __init__(self):
      json_fp = open('credentials.json')
      self.cred = json.load(json_fp)
      self.connect()
    
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
      
      

   def insertRawTweet(self, string):
      self.query("INSERT INTO tweets_raw VALUES (%s)", string)
      self.db.commit()
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

     self.db.commit()
     return

   def insertUser(self, user):
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
         0,
         0
      )
      self.query("INSERT INTO users (user_id, user_name, user_screen_name, user_location, user_description, user_url, user_followers_count, user_friends_count, user_listed_count, user_created_at, user_favourites_count, user_utc_offset, user_time_zone, user_geo_enabled, user_verified, user_statuses_count, user_lang, user_contributors_enabled, user_processed, friend_of_source) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", data)
      self.db.commit()
      return


   def loadTweet(self, tweet_id):
      """
      Tweet loader.
      @param tweet_id
      @return Populated tweet dictionary, if tweet found. "None" otherwise.
      """
      sql = "SELECT tweet_text, tweet_source, user_id, tweet_retweeted_status_id FROM tweet WHERE tweet_id = %s"
      data = (tweet_id)
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
      
   def loadUnprocessedUser(self):
      """
      Load unprocessed user
      @return Single user_id
      """
      sql = "SELECT user_id FROM users WHERE user_processed=0 LIMIT 1"
      data = ()
      c = self.query(sql, data)
      row = c.fetchone()
      if row == None:
          return None
      else:
          return row[0]
          
          
   def saveProcessedUser(self, user_id, value=1):
      sql = "UPDATE users SET user_processed=%s WHERE user_id=%s"
      data = (value, user_id)
      self.query(sql, data)
      self.db.commit()
      return

   def saveFollower(self, user_id):   
      sql = "UPDATE users SET friend_of_source=1 WHERE user_id=%s"
      data = (user_id)
      self.query(sql, data)
      self.db.commit()
      return       

   def loadUser(self, user_id):
      """
      User loader.
      @param user_id
      @return Populated user dictionary, if found. "None" otherwise.
      """
      sql = "SELECT user_name, user_screen_name, user_location, user_description, user_url, user_followers_count, user_friends_count, user_favourites_count FROM user WHERE user_id = %s"
      params = user_id
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
      self.db.commit()
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
#      `user_processed` smallint(6) DEFAULT 0
#      `friend_of_source` smallint(6) DEFAULT 0
#    );
#    (FALTA TABLEW tweets_raw)
