#Gridsearch for optimal number of hidden nodes

train = read.csv("train_fs_ig.csv")
valid = read.csv("valid_fs_ig.csv")
test = read.csv("test_fs_ig.csv")

#check number of levels for each variable, to see if any are already binary
var_levels = sapply(train, unique) #get # of levels
var_level_numbers = sapply(var_levels, length)
var_level_numbers
#foreign worker and credit risk are binary. credit risk already has values of 0 and 1
unique(train$foreign_worker) #has values 2 and 1 - want to replace with 1 and 0
train$foreign_worker = ifelse(train$foreign_worker == 2, 0L, 1L) #change 2 (no) to 0, integers
valid$foreign_worker = ifelse(valid$foreign_worker == 2, 0L, 1L) #do same for validation
test$foreign_worker = ifelse(test$foreign_worker == 2, 0L, 1L) #do same for test

library(dplyr)
#change categorical variables (that aren't already encoded 1 or 0) to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing") #leave out credit risk and foreign worker because they're already binary
train = train %>% mutate (across(all_of(cat_vars), as.factor))
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
test = test %>% mutate (across(all_of(cat_vars), as.factor))

#Standardize numeric columns - necessary for neural network
#standardize duration
mean_dur = mean(train$duration)
sd_dur= sd(train$duration)
#use values in training data to standardize validation/test data
train$duration = (train$duration - mean_dur)/sd_dur
valid$duration = (valid$duration - mean_dur)/sd_dur
test$duration = (test$duration - mean_dur)/sd_dur
#repeat for amt
mean_amt = mean(train$amount)
sd_amt = sd(train$amount)
train$amount = (train$amount - mean_amt)/sd_amt
valid$amount = (valid$amount - mean_amt)/sd_amt
test$amount = (test$amount - mean_amt)/sd_amt

#create dummy variables using model.matrix
#Note: I ran unique(train$var) and unique(valid$var) for each categorical variable to ensure that none had levels the other lacked, which would mess
# up the dummy encoding of the variables
train.d = model.matrix(credit_risk ~ ., data = train)[, -1]
valid.d = model.matrix(credit_risk ~ ., data = valid)[, -1]
test.d = model.matrix(credit_risk ~ ., data = test)[, -1]

# Combine the response and predictor variables into a single data frame
# for the neuralnet function
train.nn <- as.data.frame(cbind(train$credit_risk, train.d))
colnames(train.nn)[1] = "credit_risk"
# The same for validation and test data
valid.nn <- as.data.frame(cbind(valid$credit_risk, valid.d))
colnames(valid.nn)[1] = "credit_risk"

#Create formula to copy/paste into neuralnet.model since credit_risk ~ . won't work
# (also calling "eqn" in the model won't work either)
predictors = setdiff(colnames(train.nn), "credit_risk")
rhs = paste(predictors, collapse = " + ")
eqn = as.formula(paste("credit_risk", rhs, sep = " ~ "))

library(neuralnet)
hidden_values = c(1:21) #online sources say not to exceed # features
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

View(gridsearch_df) #so far it looks like anything > 5 overfits. the best is 2 (AUC train ~0.86, AUC valid ~ 0.71)


#Search for best stepmax (not searching for both at the same time because it will take much longer)
stepmax = c(1e4, 1e5, 1e6) #anything below 1e4 gave convergence warning
gridsearch_df2 = data.frame(
  max_steps = numeric(),
  auc_valid = numeric(),
  auc_train = numeric())
library(pROC)
for (i in stepmax){
  set.seed(2025)
  neuralnet.model = neuralnet(
    eqn,
    data = train.nn,
    hidden = 5, #trying 5 
    err.fct = "ce", #cross entropy error facet
    rep = 10,
    stepmax = i,
    linear.output = FALSE
  )
  valid.pred = predict(neuralnet.model, newdata=valid.nn, type = "response")
  roc.valid = roc(valid.nn$credit_risk, as.vector(valid.pred))
  train.pred = predict(neuralnet.model, train.nn, type = "response")
  roc.train = roc(train.nn$credit_risk, as.vector(train.pred))
  area_under_valid = auc(roc.valid)
  area_under_train = auc(roc.train)
  gridsearch_df2 = rbind(gridsearch_df, data.frame(max_steps = i, auc_valid = area_under_valid, auc_train = area_under_train))
}

View(gridsearch_df2) #they are all exactly the same


set.seed(2025)
neuralnet.model = neuralnet(
    eqn,
    data = train.nn,
    hidden = 2, 
    err.fct = "ce", #cross entropy error facet
    rep = 10,
    stepmax = 17000,
    linear.output = FALSE)

library(pROC)
#Check AUC on training data to ensure overfitting isn't happening
train.pred = predict(neuralnet.model, train.nn, type = "response")
roc.train = roc(train.nn$credit_risk, as.vector(train.pred))
auc(roc.train)

#Check AUC on validation data
valid.pred = predict(neuralnet.model, newdata=valid.nn, type = "response")
roc.valid = roc(valid.nn$credit_risk, as.vector(valid.pred))
auc(roc.valid)

#get optimal cutoff for event based on ROC
opt_cutoff = coords(roc.valid, "best", ret = "threshold") #0.913 - oddly high

library(caret)
valid.class = ifelse(valid.pred > opt_cutoff[[1]], 1, 0) #classify event based on cutoff
confusionMatrix(table(valid.class,valid$credit_risk), positive="1") #get confusion matrix values

write.csv(valid.pred, "ANN_pred_valid.csv", row.names=FALSE)

