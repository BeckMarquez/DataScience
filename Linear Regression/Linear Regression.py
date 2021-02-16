#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# In[4]:


data = pandas.read_csv('cost_revenue_clean.csv')


# In[5]:


data.describe()


# In[6]:


X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])


# In[7]:


plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3e+9)
plt.xlim(0, 4.5e+8)
plt.show


# In[8]:


regression = LinearRegression()
regression.fit(X, y)


# Slope Coefficient

# In[9]:


regression.coef_ # theta_1


# In[10]:


# Intercept
regression.intercept_


# In[13]:


plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3)
plt.plot(X, regression.predict(X), color='red', linewidth=4)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3e+9)
plt.xlim(0, 4.5e+8)
plt.show


# In[15]:


regression.score(X, y)


# In[ ]:




