#NOTE: MODEL IS OVERFITTING, NEED TO DO MORE TROUBLESHOOTING
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
             "employment_duration", "installment_rate", 
             "personal_status_sex", "other_debtors", "present_residence", 
             "property", "other_installment_plans","housing",
             "job") #leave out credit risk and foreign worker because they're already binary
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

#Oversample rare event in training data - this ended up creating overfitting so I will not use it.
#library(ROSE)
#train.balanced <- ROSE(credit_risk ~ ., data = train.nn)$data
# Check for class balance after balancing
#print(table(train.balanced$credit_risk))

#Create formula to copy/paste into neuralnet.model since credit_risk ~ . won't work
# (also calling "eqn" in the model won't work either)
predictors = setdiff(colnames(train.nn), "crendit_risk")
rhs = paste(predictors, collapse = " + ")
eqn = as.formula(paste("credit_risk", rhs, sep = " ~ "))

set.seed(2025) #ensure results are consistent
library(neuralnet)
neuralnet.model = neuralnet(
  eqn,
  data = train.nn, #try changing this to train.balanced and try again
  hidden = 6,
  err.fct = "ce", #cross entropy error facet
  rep = 10,
  stepmax = 1e6,
  linear.output = FALSE
)

#data = train.nn, hidden= 10 - AUC train of .82, AUC valid of 0.71 
#data = train.nn, hidden= 18 - AUC train of .82, AUC valid of 0.69
#data = train.nn, hidden= 7 - AUC train of .79, AUC valid of 0.73 - looks like winner so far
#data = train.nn, hidden= 6 - AUC train of .80, AUC valid of 0.70 


#AUC on training
train.pred = predict(neuralnet.model, train.nn, type="response")
roc.train = roc(train.balanced$credit_risk, as.vector(train.pred))
auc(roc.train)

#AUC on validation
valid.pred = predict(neuralnet.model, newdata=valid.nn, type = "response")
library(pROC)
roc.valid = roc(valid$credit_risk, as.vector(valid.pred))
auc(roc.valid)

#might need to try a more advanced package like keras