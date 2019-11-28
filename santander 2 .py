#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.metrics import roc_auc_score,confusion_matrix,make_scorer,classification_report,roc_curve,auc
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
import time
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import scikitplot as skplt
from scikitplot.metrics import plot_confusion_matrix,plot_precision_recall_curve
import os
SEED = 13
np.random.seed(SEED)
from sklearn import metrics
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler


# In[2]:


start_time = time.time() # loading train data
train_df=train=pd.read_csv("C:/Users/AnushaSanthosh/Desktop/train_1.csv" , nrows=120000) # the first column 'key' is not useful 
print("%s seconds" % (time.time() - start_time))


# In[3]:


start_time = time.time() # loading test data
test_df=train=pd.read_csv("C:/Users/AnushaSanthosh/Desktop/test_1.csv" , nrows=120000) # the first column 'key' is not useful 
print("%s seconds" % (time.time() - start_time))


# In[4]:


train_df.shape, test_df.shape # checking the shape


# In[42]:


train.columns # checking the col names


# In[6]:


train_df.target.value_counts()  # knowing the binary data 


# In[7]:


train_df.head(10)


# In[8]:


train_df.isna().sum().sum() #checking for train missing values


# In[9]:


test_df.head(10)  


# In[10]:


test_df.isna().sum().sum() # checking for test missing values


# In[11]:


gc.collect();
train_df.describe()  # chk statistical values in train


# In[12]:


numerical_features = train_df.columns[2:]  


# In[44]:


train_df.columns[2:]  


# In[14]:


# checking the distribution of each variables.
print('Distributions columns')
plt.figure(figsize=(30, 185))
for i, col in enumerate(numerical_features):
    plt.subplot(50, 4, i + 1)
    plt.hist(train_df[col]) 
    plt.title(col)
gc.collect();


# In[15]:


plt.figure(figsize=(20, 8)) # checkig the distribution for mean
train_df[numerical_features].median().plot('hist');
plt.title('Median Frequency');


# In[16]:


plt.figure(figsize=(20, 8))  # checkig the distribution for SD
train[numerical_features].std().plot('hist');
plt.title('Standard Deviation Frequency');


# In[17]:


plt.figure(figsize=(20, 8))  # checkig the skewness frequency
train[numerical_features].skew().plot('hist');
plt.title('Skewness Frequency');


# In[18]:


plt.figure(figsize=(20, 8))  # checking kurtosis frequency
train[numerical_features].kurt().plot('hist');
plt.title('Kurtosis Frequency');


# In[19]:


# checking the correlation of numerical freatures with  all the variables.
corr = train_df[numerical_features].corr()
s = corr.unstack().drop_duplicates()
so = s.sort_values(kind="quicksort")
so = so.drop_duplicates()

print("Top most highly positive correlated features:")
print(so[(so<1) & (so>0.5)].sort_values(ascending=False))

print()

print("Top most highly megative correlated features:")
print(so[(so < - 0.005)])


# In[20]:


train_df.shape, test_df.shape


# In[21]:


# checking correlation of numerical features with all the test variables.
corr = test_df[numerical_features].corr()
s = corr.unstack().drop_duplicates()
so = s.sort_values(kind="quicksort")
so = so.drop_duplicates()

print("Top most highly positive correlated features:")
print(so[(so<1) & (so>0.5)].sort_values(ascending=False))

print()

print("Top most highly megative correlated features:")
print(so[(so < - 0.005)])


# In[22]:


# plotting the correlation of test and train

#Correlations in train data
train_correlations=train_df[numerical_features].corr()
train_correlations=train_correlations.values.flatten()
train_correlations=train_correlations[train_correlations!=1]
#Correlations in test data
test_correlations=test_df[numerical_features].corr()
test_correlations=test_correlations.values.flatten()
test_correlations=test_correlations[test_correlations!=1]

plt.figure(figsize=(20,5))
#Distribution plot for correlations in train data
sns.distplot(train_correlations, color="Red", label="train")
#Distribution plot for correlations in test data
sns.distplot(test_correlations, color="Blue", label="test")
plt.xlabel("Correlation values found in train and test")
plt.ylabel("Density")
plt.title("Correlation distribution plot for train and test attributes")
plt.legend()


# In[23]:


# splitting the data for ML
X=train_df.drop(columns=['ID_code','target'],axis=1)
test=test_df.drop(columns=['ID_code'],axis=1)
Y=train_df['target']
cv=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
for train_index,valid_index in cv.split(X,Y):
    X_train, X_valid=X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid=Y.iloc[train_index], Y.iloc[valid_index]

print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# In[24]:


# using logistic regression

#Logistic regression model
lr_model=LogisticRegression(random_state=42)
#fitting the lr model

lr_model.fit(X_train,y_train)
#Accuracy of the model
lr_score=lr_model.score(X_train,y_train)
print('Accuracy of the lr_model :',lr_score)

#Cross validation prediction
cv_predict=cross_val_predict(lr_model,X_valid,y_valid,cv=5)
#Cross validation score
cv_score=cross_val_score(lr_model,X_valid,y_valid,cv=5)
print('cross_val_score :',np.average(cv_score))


# In[25]:


#Confusion matrix
cm=confusion_matrix(y_valid,cv_predict)
#Plot the confusion matrix
plot_confusion_matrix(y_valid,cv_predict,normalize=False,figsize=(15,8))


# In[26]:


#ROC_AUC score
roc_score=roc_auc_score(y_valid,cv_predict)
print('ROC score :',roc_score)

#ROC_AUC curve
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_valid,cv_predict)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# In[27]:


#Classification report
scores=classification_report(y_valid,cv_predict)
print(scores)


# In[28]:


#Split the training data
X_train,X_valid,y_train,y_valid=train_test_split(X,Y,random_state=42)

print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# In[29]:


#Random forest classifier
rf_model=RandomForestClassifier(n_estimators=10,random_state=42)
#fitting the model
rf_model.fit(X_train,y_train)


# In[30]:


rf_score=rf_model.score(X_train,y_train)
print('Accuracy of the rf_model :',rf_score)


# In[31]:


rf1_predict=cross_val_predict(rf_model,X_valid,y_valid,cv=5)
#Cross validation score
rf1_score=cross_val_score(rf_model,X_valid,y_valid,cv=5)
print('cross_val_score :',np.average(rf_score))


# In[32]:


cm1=confusion_matrix(y_valid,rf1_predict)
#Plot the confusion matrix
plot_confusion_matrix(y_valid,rf1_predict,normalize=False,figsize=(15,8))


# In[34]:


#ROC_AUC score
roc_score=roc_auc_score(y_valid,rf1_predict)
print('ROC score :',roc_score)

#ROC_AUC curve
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_valid,rf1_predict)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# In[35]:


#Classification report
scores=classification_report(y_valid,rf1_predict)
print(scores)


# In[36]:


#Synthetic Minority Oversampling Technique
sm = SMOTE(random_state=42, ratio=1.0)
#Generating synthetic data points
X_smote,y_smote=sm.fit_sample(X_train,y_train)
X_smote_v,y_smote_v=sm.fit_sample(X_valid,y_valid)


# In[41]:


get_ipython().run_cell_magic('time', '', "#Logistic regression model for SMOTE\nsmote=LogisticRegression(random_state=42)\n#fitting the smote model\nsmote.fit(X_smote,y_smote)\n\n#Accuracy of the model\nsmote_score=smote.score(X_smote,y_smote)\nprint('Accuracy of the smote_model :',smote_score)\n\n#Cross validation prediction\ncv_pred=cross_val_predict(smote,X_smote_v,y_smote_v,cv=5)\n#Cross validation score\ncv_score=cross_val_score(smote,X_smote_v,y_smote_v,cv=5)\nprint('cross_val_score :',np.average(cv_score))\ncm=confusion_matrix(y_smote_v,cv_pred)\n#Plot the confusion matrix\nplot_confusion_matrix(y_smote_v,cv_pred,normalize=False,figsize=(15,8))\n\n#ROC_AUC score\nroc_score=roc_auc_score(y_smote_v,cv_pred)\nprint('ROC score :',roc_score)\n\n#ROC_AUC curve\nplt.figure()\nfalse_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_pred)\nroc_auc=auc(false_positive_rate,recall)\nplt.title('Reciver Operating Characteristics(ROC)')\nplt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)\nplt.legend()\nplt.plot([0,1],[0,1],'r--')\nplt.xlim([0.0,1.0])\nplt.ylim([0.0,1.0])\nplt.ylabel('Recall(True Positive Rate)')\nplt.xlabel('False Positive Rate')\nplt.show()\nprint('AUC:',roc_auc)\n\n#Classification report\nscores=classification_report(y_smote_v,cv_pred)\nprint(scores)\n\n\n%%time\n#Predicting the model\nX_test=test_df.drop(['ID_code'],axis=1)\nsmote_pred=smote.predict(X_test)\nprint(smote_pred)")


# In[43]:


#final submission
sub_df=pd.DataFrame({'ID_code':test_df['ID_code'].values})
sub_df['Random_Forest_predict']=rf1.predict
sub_df.to_csv('submission.csv',index=False)
sub_df.head()


# In[ ]:




