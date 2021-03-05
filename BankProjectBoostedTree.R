# Load required packages

library(tidyr)
library(islr)
library(gbm)
library(dplyr)
library(caret)
library(xgboost)

# Change working directory 

setwd('C:/Users/abugh/OneDrive/Documents/R Working Directory/BankData')

# Load in data

bankData <-  read.csv('bank-full.csv', sep = ';')

# pdays is the number of days since a client was contacted via a previous campaign, the variable has a value of -1 if the client has never been contacted
# Split the variable pdays into two new variables, 'contacted' - a binary variable that takes on a value of 1 if they have been contacted previously, and 0 otherwise. 
# Create a new variable called 'days_since' which takes on 

# Filter out rows with missing observations

bankData <- bankData %>% drop_na()




# Encode categorical variables as n-1 dummy variables

dummy <- dummyVars(" ~ job + marital + education + default + housing + loan + contact + month + poutcome + y ", data = bankData)

dummy_pred = data.frame(predict(dummy, newdata = bankData))

bankData = cbind(bankData %>% select(-c('job', "marital" , "education" , "default" , "housing" , "loan" , "contact" , "month" , "poutcome")), dummy_pred)

# Encode our target variable y as a binary variable that takes on a value of 0 when the target is 'no', and 1 when the target is 'yes

bankData$yyes = (bankData$yno * -1) + 1

bankData <- bankData %>% select(-c(y, yno))

# Split the data into test and train sets, with 1/2 of all observations in the training set and half the observations in the test test

train_index <- sample(1:nrow(bankData), nrow(bankData)/2)

train <- bankData[train_index,]
test <- bankData[-train_index,]

boost.Bank <- gbm(yyes ~., data = train, distribution = 'bernoulli', n.trees = 10000, interaction.depth = 4)

summary(boost.Bank)

yhat <- predict(boost.Bank, newdata = test, n.trees = 1000)

prediction <- as.numeric(yhat > 0.5)

table(prediction, test$yyes)

err = mean(prediction != test$yyes)