#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from RE_LSP import re_lsp
from RE_TSD import re_tsd
from RE_GSD import re_gsd
from sklearn.model_selection import train_test_split

# In[10]:


#Import data
data = pd.read_csv('SFEM RESULTS.csv', header=1)
Y = data.loc[:,['y1','y2','y3']]
Y1=np.array(data.pop('y1')).reshape(-1,1)
Y2=np.array(data.pop('y2')).reshape(-1,1)
Y3=np.array(data.pop('y3')).reshape(-1,1)
X= data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# In[11]:


#Find linear regression model
pre_lsp=re_lsp.predict(X).reshape(-1,1)
pre_tsd=re_tsd.predict(X).reshape(-1,1)
pre_gsd=re_gsd.predict(X).reshape(-1,1)
x_lsp = sm.add_constant(pre_lsp)
x_tsd = sm.add_constant(pre_tsd)
x_gsd = sm.add_constant(pre_gsd)
fit_lsp=sm.OLS(Y1,x_lsp).fit()
fit_tsd=sm.OLS(Y2,x_tsd).fit()
fit_gsd=sm.OLS(Y3,x_gsd).fit()
print(fit_lsp.summary())
print(fit_tsd.summary())
print(fit_gsd.summary())


