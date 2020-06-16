#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os

# import model related libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import module to calculate model perfomance metrics
from sklearn import metrics


# In[10]:


data = pd.read_csv('Salary_Data.csv')


# In[12]:


data.head()


# In[13]:


data.tail()


# In[14]:


data.sample(7)


# In[16]:


data.shape


# In[17]:


data.dtypes


# ## Describe data statistically 

# In[18]:


data.describe()


# In[19]:


data.info()


# ## Data Cleaning

# In[20]:


data.shape


# In[21]:


data=data.drop_duplicates()


# In[22]:


data.shape


# In[24]:


data.isnull().sum()


# ## Create dependent(y) and independent(X) variables

# In[33]:


target_feature='Salary'
y=data[target_feature]
X=data.drop(target_feature,axis=1)


# In[38]:


X.shape


# In[40]:


X.head()


# In[42]:


y.shape


# In[43]:


y.head()


# ## Data visualization before training the model

# In[44]:


import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.grid()
plt.show()


# ## Splitting X and y into training and testing sets

# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.20)


# In[57]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ## Apply linear regression on train dataset 

# In[58]:


# Linear Regression Model
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)


# In[59]:


print("Intercept=",linreg.intercept_)
print("Slope=",linreg.coef_)


# ## Apply the model on test dataset to get the predicted values 

# In[60]:


y_pred = linreg.predict(X_test)


# In[61]:


y_pred


# ## Compare actual output values to the predicted values 

# In[62]:


data1=pd.DataFrame({'Actual':y_test,'Predicted':y_pred,'variance':y_test-y_pred})


# In[63]:


data1


# ## Prediction 

# In[65]:


# Predicting the salary for 1.5 years of experience 
pred=np.array([1.5]).reshape(-1,1)


# In[67]:


linreg.predict(pred)


# ## Visualization

# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

x=np.arange(1,len(y_test)+1)

plt.plot(x,y_test,label='Actual')
plt.plot(x,y_pred,label='Predicted')
plt.title("Actual Vs Predicted (Test set)")
plt.legend(loc="best")
plt.grid(True)


# In[69]:


#Visualizing the test set result 
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, linreg.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.grid()
plt.show()


# In[74]:


#Visualizing the test set result 
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, linreg.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.grid()
plt.show()


# ## Evaluation Metrics 

# In[75]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)*100
print("score",score)


# In[76]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

