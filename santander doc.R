
rm(list=ls())

library(tidyverse)
library(caret)
library(Matrix)
library(caTools)
library(randomForest)
library(glmnet)

library(moments)

if (!require("glmnet")) {
  install.packages("glmnet", repos="http://cran.rstudio.com/") 
  library("glmnet")
}

library(glmnet)

if (!require("e1071")) {
  install.packages("e1071", repos="http://cran.rstudio.com/") 
  library("e1071")
}
library(e1071)


library(pROC)

library(DMwR)

library(ROSE)
library(dplyr)




setwd("C:/Users/AnushaSanthosh/Desktop")
train_df<-read.csv("C:/Users/AnushaSanthosh/Desktop/train_1.csv",nrows=50000)
head(train_df)
dim(train_df)
str(train_df)
train_df$target<-as.factor(train_df$target)
table(train_df$target)
for (var in names(train_df)[c(3:202)]){
  target<-train_df$target
  plot<-ggplot(train_df, aes(x=train_df[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}


for (var in names(train_df)[c(103:202)]){
  target<-train_df$target
  plot<-ggplot(train_df, aes(x=train_df[[var]], fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}
test_df<-read.csv("C:/Users/AnushaSanthosh/Desktop/test_1.csv",nrows=50000)
head(test_df)
dim(test_df)
#Applying the function to find mean values per row in train and test data.
train_mean<-apply(train_df[,-c(1,2)],MARGIN=1,FUN=mean)
test_mean<-apply(test_df[,-c(1)],MARGIN=1,FUN=mean)
ggplot()+
  #Distribution of mean values per row in train data
  geom_density(data=train_df[,-c(1,2)],aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=test_df[,-c(1)],aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per row',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find mean values per column in train and test data.
train_mean<-apply(train_df[,-c(1,2)],MARGIN=2,FUN=mean)
test_mean<-apply(test_df[,-c(1)],MARGIN=2,FUN=mean)
ggplot()+
  #Distribution of mean values per column in train data
  geom_density(aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per column in test data
  geom_density(aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per column',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find standard deviation values per row in train and test data.
train_sd<-apply(train_df[,-c(1,2)],MARGIN=1,FUN=sd)
test_sd<-apply(test_df[,-c(1)],MARGIN=1,FUN=sd)
ggplot()+
  #Distribution of sd values per row in train data
  geom_density(data=train_df[,-c(1,2)],aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=test_df[,-c(1)],aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per row',title="Distribution of sd values per row in train and test dataset")

#Applying the function to find sd values per column in train and test data.
train_sd<-apply(train_df[,-c(1,2)],MARGIN=2,FUN=sd)
test_sd<-apply(test_df[,-c(1)],MARGIN=2,FUN=sd)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per column',title="Distribution of std values per column in train and test dataset")

#Applying the function to find skewness values per row in train and test data.
train_skew<-apply(train_df[,-c(1,2)],MARGIN=1,FUN=skewness)
test_skew<-apply(test_df[,-c(1)],MARGIN=1,FUN=skewness)
ggplot()+
  #Distribution of skewness values per row in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per row',title="Distribution of skewness values per row in train and test dataset")

#Applying the function to find skewness values per column in train and test data.
train_skew<-apply(train_df[,-c(1,2)],MARGIN=2,FUN=skewness)
test_skew<-apply(test_df[,-c(1)],MARGIN=2,FUN=skewness)
ggplot()+
  #Distribution of skewness values per column in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per column',title="Distribution of skewness values per column in train and test dataset")

#Applying the function to find kurtosis values per row in train and test data.
train_kurtosis<-apply(train_df[,-c(1,2)],MARGIN=1,FUN=kurtosis)
test_kurtosis<-apply(test_df[,-c(1)],MARGIN=1,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per row',title="Distribution of kurtosis values per row in train and test dataset")

#Applying the function to find kurtosis values per column in train and test data.
train_kurtosis<-apply(train_df[,-c(1,2)],MARGIN=2,FUN=kurtosis)
test_kurtosis<-apply(test_df[,-c(1)],MARGIN=2,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per column',title="Distribution of kurtosis values per column in train and test dataset")

#Finding the missing values in train data
missing_val<-data.frame(missing_val=apply(train_df,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val

#Finding the missing values in test data
missing_val<-data.frame(missing_val=apply(test_df,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val

#Correlations in train data
#convert factor to int
train_df$target<-as.numeric(train_df$target)
train_correlations<-cor(train_df[,c(2:202)])
train_correlations

#Correlations in test data
test_correlations<-cor(test_df[,c(2:201)])
test_correlations


#Split the data using CreateDataPartition
set.seed(689)
#train.index<-createDataPartition(train_df$target,p=0.8,list=FALSE)
train.index<-sample(1:nrow(train_df),0.8*nrow(train_df))
#train data
train.data<-train_df[train.index,]
#validation data
valid.data<-train_df[-train.index,]
#dimension of train data
dim(train.data)
#dimension of validation data
dim(valid.data)
#target classes in train data
table(train.data$target)
#target classes in validation data
table(valid.data$target)

#Training dataset
X_t<-as.matrix(train.data[,-c(1,2)])
y_t<-as.matrix(train.data$target)
#validation dataset
X_v<-as.matrix(valid.data[,-c(1,2)])
y_v<-as.matrix(valid.data$target)
#test dataset
test<-as.matrix(test_df[,-c(1)])
set.seed(667) # to reproduce results
lr_model <-glmnet(X_t,y_t, family = "binomial")
summary(lr_model)
set.seed(8909)
cv_lr <- cv.glmnet(X_t,y_t,family = "binomial", type.measure = "class")
cv_lr

#Minimum lambda
cv_lr$lambda.min
#plot the auc score vs log(lambda)
plot(cv_lr)

set.seed(5363)
cv_predict.lr<-predict(cv_lr,X_v,s = "lambda.min", type = "class")
cv_predict.lr

#Confusion matrix
set.seed(689)
#actual target variable
target<-valid.data$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.lr<-as.factor(cv_predict.lr)
confusionMatrix(data=cv_predict.lr,reference=target)

set.seed(892)
cv_predict.lr<-as.numeric(cv_predict.lr)
roc(response=target,predictor=cv_predict.lr,auc=TRUE,plot=TRUE)

#Random Oversampling Examples(ROSE)
set.seed(699)
train.rose <- ROSE(target~., data =train.data[,-c(1)],seed=32)$data
#target classes in balanced train data
table(train.rose$target)
valid.rose <- ROSE(target~., data =valid.data[,-c(1)],seed=42)$data
#target classes in balanced valid data
table(valid.rose$target)
set.seed(462)
lr_rose <-glmnet(as.matrix(train.rose),as.matrix(train.rose$target), family = "binomial")
summary(lr_rose)
#Cross validation prediction
set.seed(473)
cv_rose = cv.glmnet(as.matrix(valid.rose),as.matrix(valid.rose$target),family = "binomial", type.measure = "class")
cv_rose

#Minimum lambda
cv_rose$lambda.min
#plot the auc score vs log(lambda)
plot(cv_rose)

set.seed(442)
cv_predict.rose<-predict(cv_rose,as.matrix(valid.rose),s = "lambda.min", type = "class")
cv_predict.rose

#Confusion matrix
set.seed(478)
#actual target variable
target<-valid.rose$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.rose<-as.factor(cv_predict.rose)
#Confusion matrix
confusionMatrix(data=cv_predict.rose,reference=target)


#ROC_AUC score and curve
set.seed(843)
#convert to numeric
cv_predict.rose<-as.numeric(cv_predict.rose)
roc(response=target,predictor=cv_predict.rose,auc=TRUE,plot=TRUE)
#Split the training data using simple random sampling
train_index<-sample(1:nrow(train_df),0.75*nrow(train_df))
#train data
train_data<-train_df[train_index,]
#validation data
valid_data<-train_df[-train_index,]
#dimension of train and validation data
dim(train_data)
dim(valid_data)

#Training the Random forest classifier
set.seed(2732)
#convert to int to factor
train_data$target<-as.factor(train_data$target)
valid_data$target<-as.factor(valid_data$target)
#setting the mtry
mtry<-floor(sqrt(200))
#setting the tunegrid
tuneGrid<-expand.grid(.mtry=mtry)
#fitting the ranndom forest
rf<-randomForest(target~.,train_data[,-c(1)],mtry=mtry,ntree=10,importance=TRUE)
pred.rf<-predict(rf,newdata=valid_data)
cmu<-confusionMatrix(pred.rf,valid_data$target)
cmu

pred.rf<-as.numeric(pred.rf)
roc(response=valid_data$target,predictor=pred.rf,auc=TRUE,plot=TRUE)

submission<-data.frame(ID=test_df , target=pred.rf)

