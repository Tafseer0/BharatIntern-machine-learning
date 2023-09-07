#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction using Linear Regression

# # Importing Important Linraries

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation

import matplotlib.pyplot as plt 
import seaborn as sns


# # Load Winequality_Data Dataset

# In[3]:


wine =pd.read_csv("Winequality_Data.csv")


# In[4]:


wine.head(10)


# In[5]:


wine.shape


# In[6]:


wine.info()


# In[7]:


wine.describe()


# # Data cleaning

# In[8]:


sns.heatmap(wine)


# In[9]:


wine.isnull().sum()


# In[10]:


wine.duplicated().sum()


# In[11]:


wine.drop_duplicates(keep = 'first', inplace = True)
wine.duplicated().sum()


# In[12]:


wine['total sulfur dioxide'].isnull().sum()


# # Data Visualization

# In[13]:


#redwine quality range and their count 
wine['quality'].hist(color= "green",figsize = (6,5))


# # Analysis of 'quality' with respect  to other features 

# In[15]:


plt.figure(figsize=(15,15))
plt.subplot(4,3,1)
sns.barplot(x ='quality', y='fixed acidity', data = wine,palette = "winter")
plt.subplot(4,3,2)
sns.barplot(x ='quality', y='volatile acidity', data = wine,palette = "winter")
plt.subplot(4,3,3)
sns.barplot(x ='quality', y='citric acid', data = wine,palette = "winter")
plt.subplot(4,3,4)
sns.barplot(x ='quality', y='residual sugar', data = wine,palette = "winter")
plt.subplot(4,3,5)
sns.barplot(x ='quality', y='chlorides', data = wine,palette = "winter")
plt.subplot(4,3,6)
sns.barplot(x ='quality', y='free sulfur dioxide', data =wine,palette = "winter")
plt.subplot(4,3,7)
sns.barplot(x ='quality', y='total sulfur dioxide', data = wine,palette = "winter")
plt.subplot(4,3,8)
sns.barplot(x ='quality', y='density', data = wine,palette = "winter")
plt.subplot(4,3,9)
sns.barplot(x ='quality', y='pH', data = wine,palette = "winter")
plt.subplot(4,3,10)
sns.barplot(x ='quality', y='sulphates', data = wine,palette = "winter")
plt.subplot(4,3,11)
sns.barplot(x ='quality', y='alcohol', data = wine,palette = "winter")


# # Heatmeap

# In[17]:


plt.figure(figsize=(10,10))
sns.heatmap(wine.corr(), annot = True, cmap = 'Blues', fmt = '.1f')


# In[18]:


sns.distplot(wine['alcohol'])


# # Lets visualize whole data

# In[20]:


sns.pairplot(wine)


# # Feature selection

# In[22]:


# Create Classification version of target variable
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]
wine['goodquality'].value_counts()


# # Data preprocessing and model training

# In[23]:


# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
y = wine['goodquality']


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)


# In[27]:


#logistic regression
lmodel = LogisticRegression()
lmodel.fit(X_train,y_train)

y_pred = lmodel.predict(X_test)


# In[30]:


print("Accuracy Score:",accuracy_score(y_test,y_pred)*100)


# In[ ]:




