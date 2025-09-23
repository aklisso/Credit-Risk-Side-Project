# Using neural network to predict binary outcome: credit risk (good vs poor)

# Read in the data (cleaned + feature selection implemented)
train = read.csv("train_fs_ig.csv")
valid = read.csv("valid_fs_ig.csv")
test = read.csv("test_fs_ig.csv")

# Check number of levels for each variable, to see if any are already binary. Whichever ones are not will need to be
# transformed into dummy variables.
var_levels = sapply(train, unique)
var_level_numbers = sapply(var_levels, length)
var_level_numbers

# Foreign worker and credit risk are binary. Credit risk already has values of 0 and 1
unique(train$foreign_worker) #Has values 2 and 1 - want to replace with 1 and 0
train$foreign_worker = ifelse(train$foreign_worker == 2, 0L, 1L) #Change 2 (no) to 0
valid$foreign_worker = ifelse(valid$foreign_worker == 2, 0L, 1L) #Do the same for validation dataset
test$foreign_worker = ifelse(test$foreign_worker == 2, 0L, 1L) #Do same for test dataset

library(dplyr)

#Need to change categorical variables (that aren't already encoded 1/0) to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing") #Leave out credit risk and foreign worker because they're already 1/0
train = train %>% mutate (across(all_of(cat_vars), as.factor))
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
test = test %>% mutate (across(all_of(cat_vars), as.factor))

# Standardize continuous variables - necessary for neural network

#Duration
# Find mean and standard deviation
mean_dur = mean(train$duration)
sd_dur= sd(train$duration)
# Use mean/SD from training data to standardize train/validation/test data
train$duration = (train$duration - mean_dur)/sd_dur
valid$duration = (valid$duration - mean_dur)/sd_dur
test$duration = (test$duration - mean_dur)/sd_dur

# Amount
# Find mean and standard deviation
mean_amt = mean(train$amount)
sd_amt = sd(train$amount)
# Use mean/SD from training data to standardize train/validation/test data
train$amount = (train$amount - mean_amt)/sd_amt
valid$amount = (valid$amount - mean_amt)/sd_amt
test$amount = (test$amount - mean_amt)/sd_amt

#Create dummy variables using model.matrix
train.d = model.matrix(credit_risk ~ ., data = train)[, -1]
valid.d = model.matrix(credit_risk ~ ., data = valid)[, -1]
test.d = model.matrix(credit_risk ~ ., data = test)[, -1]

# Combine the response and predictor variables into a single data frame for the neuralnet function
train.nn <- as.data.frame(cbind(train$credit_risk, train.d))
colnames(train.nn)[1] = "credit_risk"
# Repeat for validation and test data
valid.nn <- as.data.frame(cbind(valid$credit_risk, valid.d))
colnames(valid.nn)[1] = "credit_risk"
test.nn <- as.data.frame(cbind(test$credit_risk, test.d))
colnames(test.nn)[1] = "credit_risk"

# Create formula to copy/paste into the neuralnet function later, since passing in "credit_risk ~ ." does not work
predictors = setdiff(colnames(train.nn), "credit_risk")
rhs = paste(predictors, collapse = " + ")
eqn = as.formula(paste("credit_risk", rhs, sep = " ~ "))

# Search for optimal number of hidden nodes
library(neuralnet)
hidden_values = c(1:14) # Online sources recommend not exceeding the number of features in the data
gridsearch_df = data.frame(
  hidden_nodes = numeric(),
  auc_valid = numeric(),
  auc_train = numeric())
library(pROC)
for (i in hidden_values){
  set.seed(2025)
  neuralnet.model = neuralnet(
    eqn,
    data = train.nn,
    hidden = i,
    err.fct = "ce", #cross entropy error facet
    rep = 10,
    stepmax = 1e6,
    linear.output = FALSE
  )
  valid.pred = predict(neuralnet.model, newdata=valid.nn, type = "response")
  roc.valid = roc(valid.nn$credit_risk, as.vector(valid.pred))
  train.pred = predict(neuralnet.model, train.nn, type = "response")
  roc.train = roc(train.nn$credit_risk, as.vector(train.pred))
  area_under_valid = auc(roc.valid)
  area_under_train = auc(roc.train)
  gridsearch_df = rbind(gridsearch_df, data.frame(hidden_values = i, auc_valid = area_under_valid, auc_train = area_under_train))
}

View(gridsearch_df) # So far it looks like anything > 5 overfits. The best is 2

library(neuralnet)
set.seed(2025)
neuralnet.model = neuralnet(
    eqn,
    data = train.nn,
    hidden = 2, 
    err.fct = "ce", #cross entropy error facet
    rep = 10,
    stepmax = 1e6,
    linear.output = FALSE)

# Check AUC on training data to ensure overfitting isn't happening
library(pROC)
train.pred = predict(neuralnet.model, train.nn, type = "response")
roc.train = roc(train.nn$credit_risk, as.vector(train.pred))
auc(roc.train)

# Check AUC/accuracy on validation data
valid.pred = predict(neuralnet.model, newdata=valid.nn, type = "response")
roc.valid = roc(valid.nn$credit_risk, as.vector(valid.pred))
auc(roc.valid)

# Get optimal cutoff for classifying event based on ROC
opt_cutoff = coords(roc.valid, "best", ret = "threshold")
library(caret)
valid.class = ifelse(valid.pred > opt_cutoff[[1]], 1, 0) #classify event based on cutoff
confusionMatrix(table(valid.class,valid$credit_risk), positive="1") #get confusion matrix values

# Find AUC on test data
test.pred = predict(neuralnet.model, newdata = test.nn, type = "response")
roc.test = roc(test.nn$credit_risk, as.vector(test.pred))
auc(roc.test)

# Get optimal cutoff for event based on ROC
opt_cutoff = coords(roc.test, "best", ret = "threshold")

# Classify event using the cutoff to get confusion matrix values
library(caret)
test.class = ifelse(test.pred > opt_cutoff[[1]], 1, 0)
cm = confusionMatrix(table(as.factor(test.class), as.factor(test$credit_risk)), positive = "1")

# Extracting precision and recall to calculate F1 score
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1

# Write validation/test predictions to CSV files to later use as inputs for stacked classifier
write.csv(valid.pred, "ANN_pred_valid.csv")
write.csv(as.vector(test.pred), "ANN_pred_test.csv")