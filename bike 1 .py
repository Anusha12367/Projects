#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import datetime
from random import randrange, uniform
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[27]:


bike_rental_data=pd.read_csv("C:/Users/AnushaSanthosh/Desktop/day.csv",index_col = 0) 

#understanding of data

bike_rental_data.shape
#It contains (731, 16)

bike_rental_data.describe()

#df_day.info()

#data  consist of Integers , Float  and Object(categorical) variables


# In[28]:


#Calculating the null values in the dataframe
missing_value = pd.DataFrame(bike_rental_data.isnull().sum())
missing_value = (missing_value/len(bike_rental_data))*100
missing_value.reset_index()

missing_value = missing_value.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
#Arranging Missing Values in Decreasing Order
missing_value = missing_value.sort_values('Missing_percentage', ascending = False)
#save output results 
missing_value.to_csv("Missing_perc.csv", index = False)
missing_value

##There is no missing value in the dataframe


# In[29]:


sns.set(style="whitegrid")
sns.boxplot(x=bike_rental_data["hum"])


# In[30]:


sns.set(style="whitegrid")
sns.boxplot(x=bike_rental_data["windspeed"])


# In[31]:


cnames = ["dteday","yr","season","mnth","workingday","weekday","weathersit","temp","atemp","hum","windspeed"]
pnames = ["temp","hum","windspeed"]


# In[32]:


#Detect & Delete Outliers
for i in pnames :
    print (i)
    q75,q25 = np.percentile(bike_rental_data.loc[:,i],[75,25])
    iqr = q75-q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print (min)
    print (max)
    
    bike_rental_data = bike_rental_data.drop(bike_rental_data[bike_rental_data.loc[:,i] < min].index)
    bike_rental_data = bike_rental_data.drop(bike_rental_data[bike_rental_data.loc[:,i] > max].index)


# In[34]:


#Converting respective variables to required data format 
bike_rental_data['dteday'] = pd.to_datetime(bike_rental_data['dteday'],yearfirst=True)
bike_rental_data['season'] = bike_rental_data['season'].astype('category')
bike_rental_data['yr'] = bike_rental_data['yr'].astype('category')
bike_rental_data['mnth'] = bike_rental_data['mnth'].astype('category')
bike_rental_data['holiday'] = bike_rental_data['holiday'].astype('category')
bike_rental_data['weekday'] = bike_rental_data['weekday'].astype('category')
bike_rental_data['workingday'] = bike_rental_data['workingday'].astype('category')
bike_rental_data['weathersit'] = bike_rental_data['weathersit'].astype('category')

bike_rental_data['temp'] = bike_rental_data['temp'].astype('float')
bike_rental_data['atemp'] = bike_rental_data['atemp'].astype('float')
bike_rental_data['hum'] = bike_rental_data['hum'].astype('float')
bike_rental_data['windspeed'] = bike_rental_data['windspeed'].astype('float')
bike_rental_data['casual'] = bike_rental_data['casual'].astype('float')
bike_rental_data['registered'] = bike_rental_data['registered'].astype('float')
bike_rental_data['cnt'] = bike_rental_data['cnt'].astype('float')


# In[35]:


##Feature selection o the basis of various features like correlation, multicollinearity.
#Correlation Plot
df_corr = bike_rental_data.loc[:,cnames]
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[36]:


#Chi Square Test of Independence
#Saving Categorical Numbers
cat_names = ["season","yr","mnth","holiday","weekday","workingday","weathersit"]


# In[37]:


from scipy.stats import chi2_contingency
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(bike_rental_data['cnt'], bike_rental_data[i]))
    print(dof)#Removing variables atemp beacuse it is highly correlated with temp,
#Removing weekday,holiday because they don;t contribute much to the independent cariable

#Removing Causal and registered becuase that's what we need to predict.

bike_rental_data = bike_rental_data.drop(['atemp','holiday','workingday','casual','registered'] ,axis=1)


# In[41]:


#Distribution of cnt
#%matplotlib inline

num_bins = 11
plt.hist(bike_rental_data['cnt'], num_bins, normed=1, facecolor='red', alpha=0.5)
plt.show()


# In[42]:



#Bike Rentals Monthly
sales_by_month = bike_rental_data.groupby('mnth').size()
print(sales_by_month)
#Plotting the Graph
plot_by_month = sales_by_month.plot(title='Monthly Sales',xticks=(1,2,3,4,5,6,7,8,9,10,11,12))
plot_by_month.set_xlabel('Months')
plot_by_month.set_ylabel('Total Bikes Rented')


# In[43]:



#Sales by Season
sales_by_weekday = bike_rental_data.groupby('season').size()
plot_by_day = sales_by_weekday.plot(title='Season Sales',xticks=(range(1,4)),rot=55)
plot_by_day.set_xlabel('season')
plot_by_day.set_ylabel('Total BIkes Rented')


# In[44]:


#Divide data into train and test
X = bike_rental_data.values[:,1:9]
Y = bike_rental_data.values[:,9]
X_train,y_train,X_test,y_test = train_test_split( X, Y, test_size = 0.2)


# In[45]:


lr_model = linear_model.LinearRegression()


# In[46]:


lr_model.fit(X_train, X_test)


# In[47]:


y_pred = lr_model.predict(y_train)


# In[48]:


errors=abs(y_pred-y_test)


# In[49]:


errors=abs(y_pred-y_test)


# In[50]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[51]:


print('RMSE: %.2f' % sqrt(mean_squared_error(y_test, y_pred)))


# In[52]:


tree = DecisionTreeRegressor().fit(X_train,X_test)


# In[53]:


prediction=tree.predict(y_train)


# In[54]:


errors=abs(prediction-y_test)


# In[55]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[56]:


print('RMSE: %.2f' % sqrt(mean_squared_error(y_test, prediction)))


# In[57]:


#RF_model = RandomForestRegressor(n_estimators = 100).fit(X_train, y_train)
RF_model = RandomForestRegressor(n_estimators = 1000, random_state = 1337)
# Train the model on training data
RF_model.fit(X_train, X_test);

# Use the forest's predict method on the test data
predictions = RF_model.predict(y_train)
# Calculate the absolute errors
errors = abs(predictions - y_test)


# In[58]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[59]:


print('RMSE: %.2f' % sqrt(mean_squared_error(y_test, predictions)))


# In[60]:


result = RF_model.predict(y_train)

