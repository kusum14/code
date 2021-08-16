import os
import time
import glob
import threading
import multiprocessing
import asyncio
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import glob
import re
import pickle
from os.path import dirname, abspath

def clean_data(msg):
        msg=re.sub('[^a-zA-Z]',' ',msg)
        msg=msg.lower()
        msg=msg.split()
        ps=PorterStemmer()
        msg=[ps.stem(word) for word in msg if not word in set(stopwords.words('english'))]
        msg=' '.join(msg)
        return msg

def predict(types,msg):
   message=[msg]
   df = pd.DataFrame(message,columns=['message']) 
   df['message']=df['message'].str.lower()
   df['message']=df['message'].apply(clean_data)
   finaldata=df['message'].values.tolist()

   if types=='auth.log':
       with open('models/authmodel','rb') as file:
           pickle_file=pickle.load(file)
       with open('models/authvector.pickel','rb') as file:
           vect_file=pickle.load(file)
       test_data=vect_file.transform(finaldata)
       pred=pickle_file.predict(test_data[0])
       prediction=pred[0]
       return prediction
      
   elif types=='daemon.log':
       with open('models/daemonmodel','rb') as file:
           pickle_file=pickle.load(file)
       with open('models/daemonvector.pickel','rb') as file:
           vect_file=pickle.load(file)
       test_data=vect_file.transform(finaldata)
       pred=pickle_file.predict(test_data[0])
       prediction=pred[0]
       return prediction
  
   elif types=='debug.log':
       with open('models/debugmodel','rb') as file:
           pickle_file=pickle.load(file)
       with open('models/debugvector.pickel','rb') as file:
           vect_file=pickle.load(file)
       test_data=vect_file.transform(finaldata)
       pred=pickle_file.predict(test_data[0])
       prediction=pred[0]
       return prediction
       
   elif types=='dmesg.log':
       with open('models/dmesgmodel','rb') as file:
           pickle_file=pickle.load(file)
       with open('models/dmesgvector.pickel','rb') as file:
           vect_file=pickle.load(file)
       test_data=vect_file.transform(finaldata)
       pred=pickle_file.predict(test_data[0])
       prediction=pred[0]
       return prediction
       
   elif types=='kern.log':
       with open('models/kernmodel','rb') as file:
           pickle_file=pickle.load(file)
       with open('models/kernvector.pickel','rb') as file:
           vect_file=pickle.load(file)
       test_data=vect_file.transform(finaldata)
       pred=pickle_file.predict(test_data[0])
       prediction=pred[0]
       return prediction

   elif types=='messages':
       with open('models/messagesmodel','rb') as file:
           pickle_file=pickle.load(file)
       with open('models/messagevector.pickel','rb') as file:
           vect_file=pickle.load(file)
       test_data=vect_file.transform(finaldata)
       pred=pickle_file.predict(test_data[0])
       prediction=pred[0]
       return prediction
       
   elif types=='syslog':
       with open('models/syslogmodel','rb') as file:
           pickle_file=pickle.load(file)
       with open('models/syslogvector.pickel','rb') as file:
           vect_file=pickle.load(file)
       test_data=vect_file.transform(finaldata)
       pred=pickle_file.predict(test_data[0])
       prediction=pred[0]
       return prediction
       
       

# In[2]:
      
       
       
      
      

async def follow(thefile):
    thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            await asyncio.sleep(0.1)
            continue
        yield line.strip()

async def main(path):
    async for x in follow(open(path)):
        if path=='/var/log/sampleClient/auth.log':
            prediction=predict("auth.log",x)
            print("auth.log"+","+x+","+str(prediction))
        if path=='/var/log/sampleClient/daemon.log':
            prediction=predict("daemon.log",x)
            print("daemon.log"+","+x+","+str(prediction))
        if path=='/var/log/sampleClient/debug.log':
            prediction=predict("debug.log",x)
            print("debug.log"+","+x+","+str(prediction))
        if path=='/var/log/sampleClient/dmesg':
            prediction=predict("dmesg",x)
            print("dmesg"+","+x+","+str(prediction))
        if path=='/var/log/sampleClient/messages':
            prediction=predict("messages",x)
            print("messages"+","+x+","+str(prediction))
        if path=='/var/log/sampleClient/kern.log':
            prediction=predict("kern.log",x)
            print("kern.log"+","+x+","+str(prediction))
        if path=='/var/log/sampleClient/syslog':
            prediction=predict("syslog",x)
            print("syslog"+","+x+","+str(prediction))
        
	



for log in ['/var/log/sampleClient/auth.log','/var/log/sampleClient/daemon.log','/var/log/sampleClient/debug.log',
             '/var/log/sampleClient/dmesg','/var/log/sampleClient/kern.log','/var/log/sampleClient/syslog','/var/log/sampleClient/messages']:
    if(os.path.exists(log)):
        asyncio.ensure_future(main(log))
    

loop = asyncio.get_event_loop()
loop.run_forever()



