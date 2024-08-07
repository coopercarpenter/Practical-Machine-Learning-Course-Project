#                                 Model Script.R
#
# 
# 
# Created: 2024-07-24
# Author: Cooper
# Purpose: Building initial models for report
#
# =============================================================================================================

# Loading Necessary Libraries and Checking Location
library(tidyverse)
library(magrittr)
library(caret)
library(here); here()

# Loading data 
training <- read_csv("data/pml-training.csv")
testing <- read_csv("data/pml-testing.csv")
# checking data
names(training)
str(training)


# ===================================== Cleaning ===============================================
# removing meta data 
training <- training[,-c(1:7)]

# removing variables that are predominantly NA (>90%)
drop_missing <- training %>% 
  summarise(across(everything(), ~sum(is.na(.))/n())) %>% 
  pivot_longer(everything(), names_to = "var", values_to = "prop_missing") %>% 
  filter(prop_missing > .9) %>% pull(var)

training <- training %>%  select(-all_of(drop_missing))

#training %>% filter(if_any(everything(), is.na))

# removing variables with near zero variance - no cases
drop_nzv <- nearZeroVar(training, names = TRUE)

# removing highly correlated variables 
cors <- cor(training[,-53])
corrplot::corrplot(cors)
            
high_cor <- findCorrelation(cors, 
                            cutoff = 0.9,
                            names = TRUE)
training <- training %>% select(-all_of(high_cor))

dim(training)

rm(drop_missing, drop_nzv, high_cor)
            

# ===================================== Splitting data ===============================================
# making a validation data set out of training data
set.seed(123)
inTrain <- createDataPartition(y = training$classe, p = .7, list = FALSE)
train <- training[inTrain,]
valid <- training[-inTrain,]
rm(inTrain)

# ===================================== Cross Validation ===============================================
cv <- trainControl(method = "cv", number = 3, verboseIter = FALSE)

# ===================================== Fitting Models ===============================================

# Decision tree
set.seed(234)
mod_tree <- train(classe ~ ., 
                  method = "rpart", 
                  preProcess = c("center", "scale"),
                  trControl = cv,
                  data = train)
plot(mod_tree)
rattle::fancyRpartPlot(mod_tree$finalModel)

pred_tree <- predict(mod_tree, valid)
confusionMatrix(factor(valid$classe), pred_tree)

# random forest 
set.seed(345)
mod_rf <- train(classe ~ ., 
                method = "rf", 
                preProcess = c("center", "scale"),
                trControl = cv,
                data = train)
plot(mod_rf)

pred_rf <- predict(mod_rf, valid)
confusionMatrix(factor(valid$classe), pred_rf)

# gradient boost
set.seed(456)
mod_gbm <- train(classe ~ .,
                 method = "gbm",
                 preProcess = c("center", "scale"),
                 trControl = cv,
                 verbose = FALSE,
                 data = train)

pred_gbm <- predict(mod_gbm, valid)
confusionMatrix(factor(valid$classe), pred_gbm)

# =====================================  Model Results ===============================================
# comparing model accuracy 
mod_results <- tibble(
  Model = c("Decision Tree", "Random Forest", "Gradient Boost"),
  Accuracy = c(confusionMatrix(factor(valid$classe), pred_tree)$overall[1],
               confusionMatrix(factor(valid$classe), pred_rf)$overall[1],
               confusionMatrix(factor(valid$classe), pred_gbm)$overall[1]),
  `Out-of-Sample Error` = 1- Accuracy)

mod_results

# ===================================== Test Data ===============================================
# applying best model (rf) to test data

pred_rf_test <- predict(mod_rf, testing)

pred_rf_test
# Looking at accuracy and out of sample error
confusionMatrix(testing$classe)

