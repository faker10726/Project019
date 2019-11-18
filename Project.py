#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# http://notepad.pw/chandanpro


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

data = pd.read_csv(r'C:\Users\KIIT_Intern\Documents\Data.csv')


from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()


# In[2]:


data.head()


# In[ ]:


data.drop('customer_id',axis=1,inplace=True)
data.drop('gender',axis=1,inplace=True)


# In[5]:


X = data.iloc[:,:-1]
Y = data.iloc[:,-1]


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)


# In[ ]:


X.loc[X['Item_Description'].str.contains('Auto Leasing'), 'Item_Description'] = '3'
X['Item_Description']=lenc.fit_transform(X['Item_Description'])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
acc = accuracy_score(Y_test,Y_pred)
print('Accuracy :',acc*100,'%')
test = pd.read_csv(r'C:\Users\nEW u\Desktop\dataset\Test.csv')

test['GL_Code'] = test['GL_Code'].str.extract('(\d+)').astype(int)
test['Vendor_Code'] = test['Vendor_Code'].str.extract('(\d+)').astype(int)
#test['Product_Category'] = test['Product_Category'].str.extract('(\d+)').astype(int)
iD = list(test['Inv_Id'])
test.drop('Inv_Id',axis=1,inplace=True)

Casscade(test]
test.loc[test['Item_Description'].str.contains('Auto Leasing'), 'Item_Description'] = '3'
#test["Item_Description"] = test["Item_Description"].str.split("- ", expand = True)[2]
#lenc = LabelEncoder()
test['Item_Description']=lenc.fit_transform(test['Item_Description'])
#test = test.iloc[:,:-1]           #Remove this later
sc_x = StandardScaler()
test = sc_x.fit_transform(test)
Y_est = classifier.predict(test)
print(Y_est)

