import tweepy
import csv
import nltk
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re
import sys
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

consumer_key = 'iDDEROs6dT9g6o3AFuPNyvUP5'
consumer_secret = 'y14XOX3L1jLzJiYfvMQiYgwPkczmK90gOA1HpkuTbzUeNnet4G'
access_key = '785123558879993857-jLlJqxvCxnCgN7crwPONUi7LPX2ZTCu'
access_secret = 'h5EhJ2dwDEIyAIpoFrGI3gI2PgbPrsx6oXppxh73vKdvz'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)

csvFile = open('inc.csv', 'a')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["Tweets","Sentiment"])
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
tok = WordPunctTokenizer()

pat1 = r'@[\w]*'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
i=1
ptweet,netweet,ntweet = 0,0,0

for tweet in tweepy.Cursor(api.search,q="Rahul Gandhi OR Congress",tweet_mode="extended",lang="en",since="2018-01-01").items(1000):
    print("Retrieving ",i,"of 1000 tweets")
    i=i+1
    tweet = tweet.full_text.translate(non_bmp_map).encode('utf-8')
    soup = BeautifulSoup(tweet,'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat,'', souped)
    stripped_ = re.sub(r'#([^\s]+)',r'\1', stripped)
    stripped_ = stripped_.replace('RT','')
    stripped_ = stripped_.replace('\\n','')
    letters_only = re.sub("[^a-zA-Z]", " ", stripped_)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    words = (" ".join(words)).strip()

    analysis = TextBlob(words)
    
    if analysis.sentiment.polarity > 0:
        polarity_tweet = 1
        ptweet+=1
    elif analysis.sentiment.polarity == 0:
        polarity_tweet = 0
        netweet+=1
    else:
        polarity_tweet = -1
        ntweet+=1
        
    csvWriter.writerow([words,polarity_tweet])

csvFile.close()
total = ptweet+netweet+ntweet
pos_percent = (ptweet/total)*100
neut_percent = (netweet/total)*100
neg_percent = (ntweet/total)*100

nature = ('Positive', 'Neutral', 'Negative')
y_pos = np.arange(len(nature))
count = [pos_percent,neut_percent,neg_percent]
 
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, nature)
plt.ylabel('Count')
plt.title('Congress Tweet Analysis')
 
plt.show()
