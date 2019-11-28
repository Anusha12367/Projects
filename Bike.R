rm(list=ls(all=T)) # clearing the R environment
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "Information",
      
      "MASS", "rpart", 'sampling', 'DataCombine', 'inTrees','fastDummies') #  loading the required libraries

lapply(x, require, character.only = TRUE)




install.packages("fastDummies" , repos="http://cran.rstudio.com/")
install.packages("corrgram" , repos="http://cran.rstudio.com/")
install.packages("inTrees" ,repos="http://cran.rstudio.com/")
install.packages("DataCombine",repos="http://cran.rstudio.com/")
install.packages("sampling",repos="http://cran.rstudio.com/")
install.packages("Information",repos="http://cran.rstudio.com/")
install.packages("dummies",repos="http://cran.rstudio.com/")
install.packages("C50",repos="http://cran.rstudio.com/")
install.packages("unbalanced",repos="http://cran.rstudio.com/")
install.packages("randomForest",repos="http://cran.rstudio.com/")
install.packages("corrgram" , repos="http://cran.rstudio.com/")
install.packages("ggplot2",repos="http://cran.rstudio.com/")
install.packages("DMwR",repos="http://cran.rstudio.com/")
install.packages("caret",repos="http://cran.rstudio.com/")
install.packages("dummies",repos="http://cran.rstudio.com/")
install.packages("inTrees",repos="http://cran.rstudio.com/")
install.packages("sampling",repos="http://cran.rstudio.com/")
install.packages("rpart" ,repos="http://cran.rstudio.com/")
install.packages("MASS",repos="http://cran.rstudio.com/")


rm(x)
bike = read.csv("C:/Users/AnushaSanthosh/Desktop/day.csv", header = T, na.strings = c(" ", "", "NA")) # loading dataset

bike_train=bike

# plotting histogram of variables.

bike_train$season=as.numeric(bike_train$season)

bike_train$mnth=as.numeric(bike_train$mnth)

bike_train$yr=as.numeric(bike_train$yr)

bike_train$holiday=as.numeric(bike_train$holiday)

bike_train$weekday=as.numeric(bike_train$weekday)

bike_train$workingday=as.numeric(bike_train$workingday)

bike_train$weathersit=as.numeric(bike_train$weathersit)

bike_train$windspeed=as.numeric(bike_train$windspeed)

par(mfrow=c(4,2))
par(mar = rep(2, 4))

hist(bike_train$season)
hist(bike_train$weather)
hist(bike_train$holiday)
hist(bike_train$workingday)
hist(bike_train$temp)
hist(bike_train$atemp)
hist(bike_train$windspeed)

# converting teh variables into required data types
bike_train$season=as.numeric(bike_train$season)

bike_train$mnth=as.numeric(bike_train$mnth)

bike_train$yr=as.factor(bike_train$yr)

bike_train$holiday=as.factor(bike_train$holiday)

bike_train$weekday=as.factor(bike_train$weekday)

bike_train$workingday=as.factor(bike_train$workingday)

bike_train$weathersit=as.factor(bike_train$weathersit)

bike_train$windspeed=as.factor(bike_train$windspeed)

bike_train=subset(bike_train,select = -c(instant,casual,registered))

d1=unique(bike_train$dteday)

df=data.frame(d1)

bike_train$dteday=as.Date(df$d1,format="%Y-%m-%d") # extracting date 

df$d1=as.Date(df$d1,format="%Y-%m-%d")

bike_train$dteday=format(as.Date(df$d1,format="%Y-%m-%d"), "%d")

bike_train$dteday=as.factor(bike_train$dteday)
str(bike_train)
missing_val = data.frame(apply(bike_train,2,function(x){sum(is.na(x))})) # checking for missing values
numeric_index = sapply(bike_train,is.numeric) #selecting only numeric



numeric_data = bike_train[,numeric_index]





cnames = colnames(numeric_data)

# checking for outliers using boxplot

for (i in 1:length(cnames))
  
{
  
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(bike_train))+ 
           
           stat_boxplot(geom = "errorbar", width = 0.5) +
           
           geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=18,
                        
                        outlier.size=1, notch=FALSE) +
           
           theme(legend.position="bottom")+
           
           labs(y=cnames[i],x="cnt")+
           
           ggtitle(paste("Box plot of count for",cnames[i])))
  
}



gridExtra::grid.arrange(gn1,gn2,ncol=3)

gridExtra::grid.arrange(gn3,gn4,ncol=2)

# plotting correlation plot to check for multicollinearity


corrgram(bike_train[,numeric_index], order = F,
         
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

bike_train = subset(bike_train,select = -c(atemp))

rmExcept("bike_train")

# model selection 
# splitting the data into traina nd test

train_index = sample(1:nrow(bike_train), 0.8 * nrow(bike_train))

train = bike_train[train_index,]

test = bike_train[-train_index,]

train

fit = rpart(cnt ~ ., data = train)

# using the DT model 

predictions_DT = predict(fit, test[,-12])

#cnames= c("dteday","season","mnth","weekday","weathersit")



which(sapply(train, function(y) nlevels(y) > 53)) 

str(train)
train$windspeed<-as.numeric(train$windspeed)
test$windspeed<-as.numeric(test$windspeed)
str(train)

# using the RF model

RF_model = randomForest(cnt ~ ., train, importance = TRUE, ntree = 200)

predictions_RF = predict(RF_model, test[,-12])

plot(RF_model)

str(train)

# checking the accuracy of the models

MAPE = function(y, yhat){
  
  mean(abs((y - yhat)/y))*100
  
}

MAPE(test[,12], predictions_DT)

MAPE(test[,12], predictions_RF)

results <- data.frame(test, pred_cnt = predictions_RF)


# submitting the most accurate model 

write.csv(results, file = 'RF output R .csv', row.names = FALSE, quote=FALSE)


