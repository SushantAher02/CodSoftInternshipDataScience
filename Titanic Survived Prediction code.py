#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


data = pd.read_csv('D:\\Codsoft Internship\\Titanic Survival Prediction\\tested.csv')


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# In[11]:


data = data.drop(columns = 'Cabin',axis=1)


# In[12]:


data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[13]:


data['Fare'].fillna(data['Fare'].mean(),inplace=True)


# In[14]:


data.isnull().sum()


# In[15]:


data.describe()


# In[17]:


data['Survived'].value_counts()


# In[19]:


data['Sex'].value_counts()


# Data Visualization

# In[20]:


sns.countplot(x = 'Survived',data=data)


# In[21]:


sns.countplot(x='Sex',data=data)


# In[22]:


sns.countplot(x='Sex',hue="Survived",data=data)


# In[23]:


sns.countplot(x='Pclass',data=data)


# In[25]:


sns.countplot(x='Pclass',hue='Survived',data=data)


# Encoding The categorical Column

# In[26]:


data['Sex'].value_counts()


# In[27]:


data['Embarked'].value_counts()


# In[29]:


#converting categorical column
data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[30]:


data.sample(5)


# Separting Feature & target

# In[31]:


x = data.drop(columns = ['PassengerId','Survived','Name','Ticket'],axis=1)#drooping the column


# In[32]:


y= data['Survived']#storing survived in y


# x.sample(5)

# In[34]:


y.sample(5)


# splitting the data into Training data and Testing data

# In[35]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state =3)


# In[36]:


print(x.shape,x_train.shape,x_test.shape)


# Model Training
# 

# Logistic Regression

# In[37]:


model = LogisticRegression()


# In[38]:


#training the logistic regression model with training data 
model.fit(x_train,y_train)


# Accuracy Score 

# In[39]:


x_train_prediction = model.predict(x_train)


# In[40]:


print(x_train_prediction)


# In[41]:


train_data_accuracy = accuracy_score(y_train,x_train_prediction)


# In[42]:


print('Accuracy Score of Training data:',train_data_accuracy)


# In[43]:


#accuracy of test data 
x_test_prediction = model.predict(x_test)


# In[44]:


print('x_prediction')


# In[46]:


print(x_test_prediction)


# In[47]:


test_data_accuracy = accuracy_score(y_test,x_test_prediction)


# In[48]:


print('Accuracy Score Of Test Data:',test_data_accuracy)


# In[ ]:




