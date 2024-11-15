#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("G:\html\jupyter\data_bank.csv")


# In[3]:


data.head()


# In[4]:


cors=data.corr()
sns.heatmap(cors)


# In[5]:


sns.boxplot(x="churn",y="age",data=data)
plt.show()


# In[6]:


sns.boxplot(x="churn",y="balance",data=data)


# In[7]:


sns.boxplot(x="churn",y="credit_score",data=data)


# In[8]:


plt.scatter(data['balance'],data['estimated_salary'])


# In[9]:


data["gender"].value_counts()


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['encoded']=encoder.fit_transform(data["gender"])
data[['churn','encoded']].head()


# In[11]:


data['encoded'].value_counts()


# In[12]:


m_values=5457
f_values=4543
m_churn,f_churn=0,0
for i in range(0,len(data)):
    if(data['churn'][i]==1 and data['encoded'][i]==1):
        m_churn+=1
    elif(data['churn'][i]==1 and data['encoded'][i]==0):
        f_churn+=1


# In[13]:


x=[m_churn,f_churn]
print(x)


# In[14]:


plt.bar(['m','f'],[m_churn,f_churn])


# In[15]:


y=data['churn']
x=data[['age','credit_score','encoded','tenure','balance','products_number','credit_card','active_member','estimated_salary']]


# In[ ]:





# In[16]:


x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
scaling.fit(x_tr)


# In[17]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_tr,y_tr)
model.score(x_ts,y_ts)


# In[18]:


model.coef_
#x=data[['age','credit_score','encoded','tenure','balance','products_number','credit_card','active_member','estimated_salary']]


# In[ ]:


from sklearn.pipeline import make_pipeline
test_score=[]
for lam in np.arange(0.01,100,0.1):
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1/lam))
    pipe.fit(x_tr,y_tr)
    scores=pipe.score(x_ts,y_ts)
    test_score.append(scores)


# In[ ]:


plt.plot(test_score)


# In[ ]:


np.argmax(test_score)


# In[31]:


x_tr.head()


# In[ ]:





# In[ ]:


l_best=0.01*624*0.1

