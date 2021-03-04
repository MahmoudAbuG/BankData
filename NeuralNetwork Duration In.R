# Load required packages

library(tidyr)
library(ISLR)
library(gbm)
library(dplyr)
library(caret)
library(xgboost)
library(keras)
library(tensorflow)


install_tensorflow(version = 'gpu')

# Change working directory 

setwd('C:/Users/abugh/Documents/R/Machine Learning Project/BankData')

# Load in data

bankData <-  read.csv('bank-additional-full.csv', sep = ';')


# Filter out rows with missing observations

bankData <- bankData %>% drop_na()



# Encode categorical variables as n-1 dummy variables

dummy <- dummyVars(" ~ job + marital + education + default + housing + loan + contact + month + poutcome + y + day_of_week ", data = bankData)

dummy_pred = data.frame(predict(dummy, newdata = bankData))

bankData = cbind(bankData %>% select(-c('job', "marital" , "education" , "default" , "housing" , "loan" , "contact" , "month" , "poutcome", 'day_of_week')), dummy_pred)

# Encode our target variable y as a binary variable that takes on a value of 0 when the target is 'no', and 1 when the target is 'yes

bankData$y.yes = (bankData$y.no * -1) + 1

bankData <- bankData %>% select(-c(y, y.no, pdays))

# Set seed for reproducibility

set.seed(1)

# Split the data into test and train sets, with 1/2 of all observations in the training set and half the observations in the test test

train_index <- sample(1:nrow(bankData), nrow(bankData)/2)

train <- bankData[train_index,]
test <- bankData[-train_index,]

# Generate the train and test labels for the response variable

train_labels = train$y.yes
test_labels = test$y.yes

# Remove response variable from train and test arrays

train <- train %>% select(-c(y.yes))
test <- test %>% select(-c(y.yes))


# Recast the training and test labels as categorical variables

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)



# Rescale data using the mean and standard deviation of the training data

mean <- apply(train, 2, mean)
std <- apply(train, 2, sd)

train <-  scale(train, center = mean, scale = std)
test <- scale(test, center = mean, scale = std)

# Reshape the data into the expected tensor shape for the specified network

train <- keras::array_reshape(train, c(20594, 62))
test <- keras::array_reshape(test, c(20594, 62))


# Specify a model as a function allowing the dropout parameter to be changed more easily
# Model consists of two hidden layers with potential dropout between each layer, activation functions are ReLU
# Final layer uses a softmax function to assign a probability 

build_model <- function(dropout = 0) {
  
  model <- keras::keras_model_sequential()
  
  model %>% 
    layer_dense(units = 100, activation = 'relu', input_shape = ncol(train)) %>% 
    layer_dropout(dropout) %>% 
    layer_dense(units = 100, activation = 'relu') %>%
    layer_dropout(dropout) %>% 
    layer_dense(units = 2, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'rmsprop', 
    loss = 'binary_crossentropy', 
    metrics = c('accuracy')
  )  
  model
}

epochs <-  50

model <- build_model()

# Print a dot for every completed epoch to track progress

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs){
    if (epoch %% 80 == 0 ) cat("\n")
    cat(".")
  }
)

# Run the model

fitted_model <- model %>% fit(train, train_labels, 
                              epochs = epochs, 
                              validation_split = 0.2, 
                              verbose = 0, 
                              callbacks = list(print_dot_callback)
                              
)

# Visualize the training progress

plot(fitted_model) + ggtitle('Duration Included and No Dropout')

# Evaluate the model

model %>% evaluate(test, test_labels, verbose = 0)


# Try and use the model with dropout = 0.4

model_dropout <- build_model(0.4)

fitted_model_dropout <- model_dropout %>% fit(train, train_labels, 
                                              epochs = epochs, 
                                              validation_split = 0.2, 
                                              verbose = 0, 
                                              callbacks = list(print_dot_callback)
                                              
)

plot(fitted_model_dropout) + ggtitle('Duration Included and Dropout = 0.4')


# Run no dropout model with one epoch

epochs <- 1


fitted_model_final <- model %>% fit(train, train_labels, 
                                    epochs = epochs, 
                                    validation_split = 0.2, 
                                    verbose = 0, 
                                    callbacks = list(print_dot_callback)
                                    
)

model %>% evaluate(test, test_labels, verbose = 0)

# Run dropout model with one epoch

fitted_model_dropout_final <- model_dropout %>% fit(train, train_labels, 
                                                    epochs = epochs, 
                                                    validation_split = 0.2, 
                                                    verbose = 0, 
                                                    callbacks = list(print_dot_callback)
)

model_dropout %>% evaluate(test, test_labels, verbose = 0)  

