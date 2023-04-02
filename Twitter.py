#!/usr/bin/env python
# coding: utf-8

# In[9]:


import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import getpass
from selenium.webdriver.chrome.service import service


# In[10]:


from selenium.webdriver.chrome.service import Service

s = Service("C:\Program Files\Driver\chromedriver.exe")

# PATH = "C:\Program Files\Driver\chromedriver.exe"
driver = webdriver.Chrome(service=s)
driver.get("https://twitter.com/i/flow/login")
# driver.maximize_window()
sleep(3)


# In[11]:


my_user = "hari1shukla"
# my_pass = getpass.getpass()
my_pass = "HPn6BV8w3DtdcGS"


# In[12]:


search_item = "Sundar Pichai"


# In[13]:


user_id = driver.find_element(By.XPATH,"//input[@type='text']")
user_id.send_keys(my_user)
user_id.send_keys(Keys.ENTER)



# In[15]:


password = driver.find_element(By.XPATH,"//input[@type='password']")
password.send_keys(my_pass)
password.send_keys(Keys.ENTER)



# In[16]:


search_box = driver.find_element(By.XPATH,"//input[@data-testid='SearchBox_Search_Input']")
search_box.send_keys(search_item)
search_box.send_keys(Keys.ENTER)


# In[17]:


all_tweets = set()


tweets = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
while True:
    for tweet in tweets:
        all_tweets.add(tweet.text)
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    sleep(3)
    tweets = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
    if len(all_tweets)>20:
        break


# In[18]:


all_tweets = list(all_tweets)
all_tweets[0]


# In[19]:


import pandas as pd
pd.options.display.max_colwidth = 1000
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# In[20]:


stp_words = stopwords.words('english')
print(stp_words)


# In[21]:


df = pd.DataFrame(all_tweets,columns=['tweets'])
df.head()


# In[22]:


one_tweet=df.iloc[0]['tweets']
one_tweet


# In[23]:


from textblob import TextBlob
from wordcloud import WordCloud

def TweetCleaning(tweet):
    cleanTweet = re.sub(r"@[a-zA-Z0-9]+","",tweet)
    cleanTweet = re.sub(r"#[a-zA-Z0-9\s]+","",cleanTweet)
    cleanTweet = ' '.join(word for word in cleanTweet.split() if word not in stp_words)
    return cleanTweet

def calPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

def calSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

def segmentation(tweet):
    if tweet > 0:
        return "positive"
    if tweet == 0:
        return "neutral"
    else:
        return "negative"


# In[24]:


df['cleanedTweets'] = df['tweets'].apply(TweetCleaning)
df['tPolarity'] = df['cleanedTweets'].apply(calPolarity)
df['tSubjectivity'] = df['cleanedTweets'].apply(calSubjectivity)
df['segmentation'] = df['tPolarity'].apply(segmentation)
df.head()


# In[25]:


df.pivot_table(index=['segmentation'],aggfunc={'segmentation':'count'})



# In[26]:


df.sort_values(by=['tPolarity'],ascending=True).head(3)


# In[27]:


df.sort_values(by=['tPolarity'],ascending=False).head(3)
#Positive Tweets Only


# In[28]:


df[df.tPolarity==0]


# In[29]:


import matplotlib.pyplot as plt

consolidated = ' '.join(word for word in df['cleanedTweets'])

wordCloud = WordCloud(width=400, height=200, random_state=20, max_font_size=119).generate(consolidated)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[76]:


positive = round(len(df[df.segmentation == 'positive'])/len(df)*100,1)
negative = round(len(df[df.segmentation == 'negative'])/len(df)*100,1)
neutral = round(len(df[df.segmentation == 'neutral'])/len(df)*100,1)

responses = [positive, negative, neutral]
responses

response = {'resp': ['Positive Tweets', 'Negative Tweets', 'Neutral Tweets'], 'Percentage':[positive, negative, neutral]}
pd.DataFrame(response)


# In[30]:


df.groupby('segmentation').count()


# In[31]:


import seaborn as sns


# In[32]:


plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
sns.scatterplot(data=df, x='tPolarity',y='tSubjectivity',s=100,hue='segmentation')


# In[34]:


sns.countplot(data=df,x='segmentation')


# In[ ]:




