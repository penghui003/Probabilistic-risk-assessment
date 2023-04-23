#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import numpy as np
import pandas as pd
from RE_LSP import re_lsp
from RE_TSD import re_tsd
from RE_GSD import re_gsd
from Linear_Regression import fit_lsp, fit_tsd,fit_gsd
from sklearn.model_selection import train_test_split
from math import sqrt

# In[4]:
data = pd.read_csv("SFEM RESULTS.csv", header=1)
Y = data.loc[:,['y1','y2','y3']]
X= data.drop(['y1','y2','y3'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
n=10000
lsp_max=100
tsd_min=0
tsd_max=15
gsd_min=0
gsd_max=30
#%%
def mc(x): 
    m=x.shape[0]
    ci_lsp=np.zeros((m,2))
    ci_tsd=np.zeros((m,2))
    ci_gsd=np.zeros((m,2))
    ci_pi_lsp=np.zeros((m,2))
    ci_pi_tsd=np.zeros((m,2))
    ci_pi_gsd=np.zeros((m,2))
    mean_lsp=np.zeros((m,1))
    mean_tsd=np.zeros((m,1))
    mean_gsd=np.zeros((m,1))
    p_lsp=np.zeros((m,1))
    p_tsd=np.zeros((m,1))
    p_gsd=np.zeros((m,1))
    p_pi_lsp=np.zeros((m,1))
    p_pi_tsd=np.zeros((m,1))
    p_pi_gsd=np.zeros((m,1))
    for i in range (0,m):
        x1_0=x[i,0] #Friction Angle
        x2_0=x[i,1] #Density
        x3_0=x[i,2] #Cohession
        x4_0=x[i,3] #Young's Modulus
        x5_0=x[i,4] #Possion's Ratio
        x6_0=x[i,5] #Depth of Tunnel1
        x7_0=x[i,6] #Radius
        x8_0=x[i,7] #Excavation Length
        x9_0=x[i,8] #Cover of Tunnel2
        x10_0=x[i,9] #Distance beteween tunnels
        x11_0=x[i,10]
        x12_0=x[i,11]
        x13_0=x[i,12]
        x14_0=x[i,13]
        x15_0=x[i,14]
        x16_0=x[i,15]
        
        mu_x3=x3_0
        sigma_x3=0.1*mu_x3
        mu_x4=x4_0
        sigma_x4=0.1*x4_0       
        mu_x5=x5_0
        sigma_x5=0.1*mu_x5
        mu_x6=x6_0
        sigma_x6=0.1*x6_0       
        mu_x7=x7_0
        sigma_x7=0.1*mu_x7
        mu_x8=x8_0
        sigma_x8=0.1*x8_0
        mu_x9=x9_0
        sigma_x9=0.1*mu_x9
        mu_x10=x10_0
        sigma_x10=0.1*x10_0 
        mu_x11=x11_0
        sigma_x11=0.1*mu_x11
        mu_x12=x12_0
        sigma_x12=0.1*x12_0
        mu_x13=x13_0
        sigma_x13=0.1*mu_x13
        mu_x14=x14_0
        sigma_x14=0.1*x14_0
        mu_x15=x15_0
        sigma_x15=0.1*x15_0
        mu_x16=x16_0
        sigma_x16=0.1*mu_x16
        
        x1=x1_0*np.ones((n,1))
        x2=x2_0*np.ones((n,1))
        x3=np.random.normal(mu_x3, sigma_x3, n).reshape(-1,1)
        x4=np.random.normal(mu_x4, sigma_x4, n).reshape(-1,1)
        x5=np.random.normal(mu_x5, sigma_x5, n).reshape(-1,1)
        x6=np.random.normal(mu_x6, sigma_x6, n).reshape(-1,1)
        x7=np.random.normal(mu_x7, sigma_x7, n).reshape(-1,1)
        x8=np.random.normal(mu_x8, sigma_x8, n).reshape(-1,1)
        x9=np.random.normal(mu_x9, sigma_x9, n).reshape(-1,1)
        x10=np.random.normal(mu_x10, sigma_x10, n).reshape(-1,1)
        x11=np.random.normal(mu_x11, sigma_x11, n).reshape(-1,1)
        x12=np.random.normal(mu_x12, sigma_x12, n).reshape(-1,1)
        x13=np.random.normal(mu_x13, sigma_x13, n).reshape(-1,1)
        x14=np.random.normal(mu_x14, sigma_x14, n).reshape(-1,1)
        x15=np.random.normal(mu_x15, sigma_x15, n).reshape(-1,1)
        x16=np.random.normal(mu_x16, sigma_x16, n).reshape(-1,1)

        var=pd.DataFrame(np.hstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16)),
                         columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16']) #Define input parameters with uncertainty
    
        
        pre_lsp=re_lsp.predict(var)
        pre_tsd=re_tsd.predict(var)
        pre_gsd=re_gsd.predict(var)
        mean_lsp[i]=np.mean(pre_lsp)
        mean_tsd[i]=np.mean(pre_tsd)
        mean_gsd[i]=np.mean(pre_gsd)
        ci_lsp[i]=np.mean(pre_lsp)-2.33*np.std(pre_lsp)/sqrt(n),np.mean(pre_lsp)+2.33*np.std(pre_lsp)/sqrt(n)
        ci_tsd[i]=np.mean(pre_tsd)-2.33*np.std(pre_tsd)/sqrt(n),np.mean(pre_tsd)+2.33*np.std(pre_tsd)/sqrt(n)
        ci_gsd[i]=np.mean(pre_gsd)-2.33*np.std(pre_gsd)/sqrt(n),np.mean(pre_gsd)+2.33*np.std(pre_gsd)/sqrt(n)
        
        ones=np.ones((n,1))
        est_lsp=fit_lsp.get_prediction(np.hstack((ones,pre_lsp.reshape(-1,1)))).summary_frame(alpha=0.1)
        est_tsd=fit_tsd.get_prediction(np.hstack((ones,pre_tsd.reshape(-1,1)))).summary_frame(alpha=0.1)
        est_gsd=fit_gsd.get_prediction(np.hstack((ones,pre_gsd.reshape(-1,1)))).summary_frame(alpha=0.1)
        PI_l_lsp, PI_u_lsp = est_lsp["obs_ci_lower"],est_lsp["obs_ci_upper"]
        PI_l_tsd, PI_u_tsd = est_tsd["obs_ci_lower"],est_tsd["obs_ci_upper"]
        PI_l_gsd, PI_u_gsd = est_gsd["obs_ci_lower"],est_gsd["obs_ci_upper"]  
        
        ci_pi_lsp[i]=np.mean(PI_l_lsp)-2.33*np.std(pre_lsp)/sqrt(n),np.mean(PI_u_lsp)+2.33*np.std(pre_lsp)/sqrt(n)
        ci_pi_tsd[i]=np.mean(PI_l_tsd)-2.33*np.std(pre_tsd)/sqrt(n),np.mean(PI_u_tsd)+2.33*np.std(pre_tsd)/sqrt(n)
        ci_pi_gsd[i]=np.mean(PI_l_gsd)-2.33*np.std(pre_gsd)/sqrt(n),np.mean(PI_u_gsd)+2.33*np.std(pre_gsd)/sqrt(n)
        
        if ci_lsp[i,0]>lsp_max:
            p_lsp[i]=0
        elif ci_lsp[i,1]<lsp_max:
            p_lsp[i]=1
        else:
            p_lsp[i]=(lsp_max-ci_lsp[i,0])/(ci_lsp[i,1]-ci_lsp[i,0])
            
        if ci_tsd[i,1]<tsd_max and ci_tsd[i,0]>tsd_min:
            p_tsd[i]=1
        elif tsd_max>ci_tsd[i,0]>tsd_min and ci_tsd[i,1]>tsd_max:
            p_tsd[i]=(tsd_max-ci_tsd[i,0])/(ci_tsd[i,1]-ci_tsd[i,0])
        elif ci_tsd[i,0]<tsd_min and tsd_min<ci_tsd[i,1]<tsd_max:
            p_tsd[i]=(tsd_min-ci_tsd[i,1])/(ci_tsd[i,0]-ci_tsd[i,1])
        elif ci_tsd[i,1]>tsd_max and ci_tsd[i,0]<tsd_min:
            p_tsd[i]=(tsd_max-tsd_min)/(ci_tsd[i,1]-ci_tsd[i,0])
        else:
            p_tsd[i]=0
    
        if ci_gsd[i,1]<gsd_max and ci_gsd[i,0]>gsd_min:
            p_gsd[i]=1
        elif gsd_max>ci_gsd[i,0]>gsd_min and ci_gsd[i,1]>gsd_max:
            p_gsd[i]=(gsd_max-ci_gsd[i,0])/(ci_gsd[i,1]-ci_gsd[i,0])
        elif ci_gsd[i,0]<gsd_min and gsd_min<ci_gsd[i,1]<gsd_max:
            p_gsd[i]=(gsd_min-ci_gsd[i,1])/(ci_gsd[i,0]-ci_gsd[i,1])
        elif ci_gsd[i,1]>gsd_max and ci_gsd[i,0]<gsd_min:
            p_gsd[i]=(gsd_max-gsd_min)/(ci_gsd[i,1]-ci_gsd[i,0])
        else:
            p_gsd[i]=0
            
        if ci_pi_lsp[i,0]>lsp_max:
            p_pi_lsp[i]=0
        elif ci_pi_lsp[i,1]<lsp_max:
            p_pi_lsp[i]=1
        else:
            p_pi_lsp[i]=(lsp_max-ci_pi_lsp[i,0])/(ci_pi_lsp[i,1]-ci_pi_lsp[i,0])
            
        if ci_pi_tsd[i,1]<tsd_max and ci_pi_tsd[i,0]>tsd_min:
            p_pi_tsd[i]=1
        elif tsd_max>ci_pi_tsd[i,0]>tsd_min and ci_pi_tsd[i,1]>tsd_max:
            p_pi_tsd[i]=(tsd_max-ci_pi_tsd[i,0])/(ci_pi_tsd[i,1]-ci_pi_tsd[i,0])
        elif ci_pi_tsd[i,0]<tsd_min and tsd_min<ci_pi_tsd[i,1]<tsd_max:
            p_pi_tsd[i]=(tsd_min-ci_pi_tsd[i,1])/(ci_pi_tsd[i,0]-ci_pi_tsd[i,1])
        elif ci_pi_tsd[i,1]>tsd_max and ci_pi_tsd[i,0]<tsd_min:
            p_pi_tsd[i]=(tsd_max-tsd_min)/(ci_pi_tsd[i,1]-ci_pi_tsd[i,0])
        else:
            p_pi_tsd[i]=0
    
        if ci_pi_gsd[i,1]<gsd_max and ci_pi_gsd[i,0]>gsd_min:
            p_pi_gsd[i]=1
        elif gsd_max>ci_pi_gsd[i,0]>gsd_min and ci_pi_gsd[i,1]>gsd_max:
            p_pi_gsd[i]=(gsd_max-ci_pi_gsd[i,0])/(ci_pi_gsd[i,1]-ci_pi_gsd[i,0])
        elif ci_pi_gsd[i,0]<gsd_min and gsd_min<ci_pi_gsd[i,1]<gsd_max:
            p_pi_gsd[i]=(gsd_min-ci_pi_gsd[i,1])/(ci_pi_gsd[i,0]-ci_pi_gsd[i,1])
        elif ci_pi_gsd[i,1]>gsd_max and ci_pi_gsd[i,0]<gsd_min:
            p_pi_gsd[i]=(gsd_max-gsd_min)/(ci_pi_gsd[i,1]-ci_pi_gsd[i,0])
        else:
            p_pi_gsd[i]=0
             
    return mean_lsp,mean_tsd,mean_gsd,ci_lsp, ci_tsd, ci_gsd,p_lsp,p_tsd,p_gsd,ci_pi_lsp,ci_pi_tsd,ci_pi_gsd,p_pi_lsp,p_pi_tsd,p_pi_gsd

#%%
mean_lsp,mean_tsd,mean_gsd,ci_lsp, ci_tsd, ci_gsd,p_lsp,p_tsd,p_gsd,ci_pi_lsp,ci_pi_tsd,ci_pi_gsd,p_pi_lsp,p_pi_tsd,p_pi_gsd=mc(X.values)
est_lsp,est_tsd,est_gsd=re_lsp.predict(X).reshape(-1,1),re_tsd.predict(X).reshape(-1,1),re_gsd.predict(X).reshape(-1,1)
mc_result=np.concatenate((est_lsp,est_tsd,est_gsd,mean_lsp,mean_tsd,mean_gsd,ci_lsp, ci_tsd, ci_gsd,p_lsp,p_tsd,p_gsd,ci_pi_lsp,ci_pi_tsd,ci_pi_gsd,p_pi_lsp,p_pi_tsd,p_pi_gsd),axis=1)
pd.DataFrame(mc_result,columns=['est_lsp','est_tsd','est_gsd','mean_lsp','mean_tsd','mean_gsd','ci_lsp_l','ci_lsp_u','ci_tsd_l','ci_tsd_u','ci_gsd_l','ci_gsd_u','p_lsp','p_tsd','p_gsd',
                                'ci-pi_lsp_l','ci-pi_lsp_u','ci-pi_tsd_l','ci-pi_tsd_u','ci-pi_gsd_l','ci-pi_gsd_u','p-pi_lsp','p-pi_tsd','p-pi_gsd']).to_csv("MC-PI_Results_a=0.1.csv")
#%%