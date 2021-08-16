#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
with open('pickle_model','rb') as file:
    pickle_file=pickle.load(file)
    
with open('vector.pickel','rb') as file:
    vect_file=pickle.load(file)
    


# In[2]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


import pandas as pd
import glob
import re
import pickle
from os.path import dirname, abspath
#parentdir = dirname(abspath('labeldata.py'))

parentdir = dirname(dirname(abspath('auth.py')))
parentdir


# In[4]:


def read_csv(file):
    df=pd.read_csv(file)
    return df

def readdata(dataset):
    path=parentdir+'/verify/'+dataset

    csv_files = glob.glob(os.path.join(path, "*.csv"))

    df_files = (pd.read_csv(f) for f in csv_files)
    df  = pd.concat(df_files, ignore_index=True)
    #df.groupby(['label']).size()
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


    
def clean_data(msg):
        msg=re.sub('[^a-zA-Z]',' ',msg)
        msg=msg.lower()
        msg=msg.split()
        ps=PorterStemmer()
        msg=[ps.stem(word) for word in msg if not word in set(stopwords.words('english'))]
        msg=' '.join(msg)
        return msg

def remove_duplicates(df):
    finaldf=df.drop_duplicates(subset={"message"},keep='first',inplace=False)
    return finaldf

def bagofwords(df,lst):
    cv=CountVectorizer(ngram_range=(1,1),max_features=4648)
    X=cv.fit_transform(lst).toarray()
    #y=df.iloc[:,1].values
    return X


# In[15]:


authlist=[]
logfile=parentdir+'/log_datasets/casper-rw/logs/auth.log'
with open(logfile) as f:
    for line in f:
        pass
    last_line = line


# In[16]:


type(last_line)


# In[17]:


lastline=[last_line]
lastline


# In[24]:


dfauth = pd.DataFrame(lastline,columns=['message']) 
dfauth['message']=dfauth['message'].str.lower()
dfauth['message']=dfauth['message'].apply(clean_data)
dfauth


# In[26]:


finaldf=remove_duplicates(dfauth)
finaldata=finaldf['message'].values.tolist()
X=bagofwords(finaldf,finaldata)


# In[41]:


test_data=vect_file.transform(finaldata)


# In[42]:


test_data


# In[43]:


pred=pickle_file.predict(test_data[0])


# In[39]:


prediction=pred[0]


# In[40]:


prediction


# In[86]:





# In[ ]:




