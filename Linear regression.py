#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[10]:


df=pd.read_csv("C:\\Users\\User\\IdeaProjects\\Salary_dataset.csv")
df


# In[14]:


df.head(10)


# In[15]:


df.size


# In[17]:


df.tail(10)


# In[18]:


df.shape


# In[20]:


df.info()


# In[23]:


df.isnull().sum()


# # check for outliers

# In[24]:


import matplotlib.pyplot as plt


# In[55]:


df.head(10).sort_values(by='YearsExperience')


# In[56]:


s=df.Salary.head(10)
y=df.YearsExperience.head(10).sort_values()
plt.plot(s,y)
plt.ylabel('Years of experience')
plt.xlabel('Salary')
plt.show()


# # Linear regression

# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[58]:


help(LinearRegression)


# In[60]:


reg=LinearRegression(fit_intercept=True)


# In[79]:


x=head.loc[:,['YearsExperience']].values
y=head.loc[:,'Salary'].values
y.shape


# In[80]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[81]:


reg.fit(x_train,y_train)


# In[91]:


x_train,y_train,x_test


# In[85]:


x_t=reg.predict(x_test)
x_t


# In[92]:


y_test


# In[96]:


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print(r2_score(y_test,x_t))
print(mean_squared_error(y_test,x_t))
print(mean_absolute_error(y_test,x_t))


# In[ ]:




