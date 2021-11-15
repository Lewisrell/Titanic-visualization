#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd


# In[17]:


titanic = pd.read_csv("Desktop/train (1).csv")


# In[18]:


titanic.head()


# In[19]:


titanic.info()


# In[20]:


titanic.describe()


# In[21]:


titanic.duplicated()


# In[22]:


titanic.duplicated().sum()


# In[23]:


titanic.isnull().mean()


# In[24]:


titanic['Age'].fillna(titanic['Age'].mean(), inplace = True)


# In[25]:


titanic['Age'] = np.ceil(titanic['Age'].values).astype(int)


# In[26]:


titanic['Age'].values


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


titanic['Survived'].value_counts()


# In[29]:


plt.figure(figsize=(10,5))


# In[30]:


sns.countplot(x='Survived', data=titanic, palette='winter').set(ylabel='No of people', title='Survivor value counts')


# In[ ]:




