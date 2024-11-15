#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[4]:


data=pd.read_csv('Disease_data.csv')


# In[5]:


data


# In[6]:


data.shape


# In[7]:


data['Disease'].unique()


# In[8]:


data['Blood Pressure'].unique()


# In[9]:


data['Cholesterol Level'].unique()


# In[10]:


data['Outcome Variable'].unique()


# In[11]:


data.columns


# In[12]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data['Fever'] = LE.fit_transform(data['Fever'])
data['Cough'] = LE.fit_transform(data['Cough'])
data['Fatigue'] = LE.fit_transform(data['Fatigue'])
data['Difficulty Breathing'] = LE.fit_transform(data['Difficulty Breathing'])
data['Gender'] = LE.fit_transform(data['Gender'])
data['Blood Pressure'] = LE.fit_transform(data['Blood Pressure'])
data['Cholesterol Level'] = LE.fit_transform(data['Cholesterol Level'])
data['Outcome Variable'] = LE.fit_transform(data['Outcome Variable'])


# In[13]:


category_counts = data['Disease'].value_counts()
data['Disease_freq'] = data['Disease'].map(category_counts)
data= data.drop(columns='Disease',axis=1)
data.head()


# In[17]:


import seaborn as sns
sns.pairplot(data,hue='Fever')
plt.show()


# In[17]:


import seaborn as sn
corr_matrix = data.corr()
plt.figure(figsize = (10,7))
sns.heatmap(corr_matrix, annot=True)
plt.xlabel('Heatmap')
plt.ylabel('Disease')


# In[18]:


X = data.drop(columns='Outcome Variable',axis=1)
X


# In[19]:


y = data[['Outcome Variable']]
y


# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
y_train_1d = np.ravel(y_train)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
RNF = RandomForestClassifier()
from sklearn.model_selection import cross_val_score
from hyperopt import fmin,Trials,tpe,hp,STATUS_OK


# In[22]:


pip install hyperopt


# In[23]:


Space = {
    'n_estimators' : hp. quniform('n_estimators',50,500,50),
    'criterion' : hp.choice('criterion',['gini', 'entropy', 'log_loss']),
    'max_depth' : hp.quniform('max_depth',1,10,1),
    'max_features' : hp.choice('max_features',['sqrt', 'log2', None])
}   


# In[24]:


def Bayesian(Space):
  RNF = RandomForestClassifier(
      n_estimators = int(Space['n_estimators']),
      criterion = Space['criterion'],
      max_depth = int(Space['max_depth']),
      max_features = Space['max_features']
  )
  accuracy = cross_val_score(RNF,X_train,y_train_1d,cv=5).mean()
  return{'loss':-accuracy,'status':STATUS_OK}


# In[25]:


trials = Trials()
Best = fmin(fn=Bayesian,space=Space,algo=tpe.suggest,max_evals = 200,trials=trials)
Best


# In[26]:


RNF = RandomForestClassifier(
      n_estimators = 400,
      criterion = 'log_loss',
      max_depth = 9,
      max_features = 'sqrt'
  )
RNF.fit(X_train,y_train_1d)


# In[27]:


y_hat = RNF.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_hat))


# In[28]:


print(classification_report(y_test,y_hat))


# In[31]:


import matplotlib.pyplot as plt
categories = ['Fever','Cough','Fatigue','Difficulty Breathing','Blood Pressure','Cholesterol Level']
values = [25, 30, 35, 20, 40, 25 ]
plt.figure(figsize = (8,8))
plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
plt.title('Disease Prediction')
plt.show()

