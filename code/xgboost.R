# Using extreme gradient boosting (xgboost) to predict binary outcome: credit risk (good vs poor)

# Read in the data (cleaned + feature selection implemented)
train = read.csv("train_fs_ig.csv")
valid = read.csv("valid_fs_ig.csv")
test = read.csv("test_fs_ig.csv")


# Ensure all binary variables have levels 0 and 1
unique(train$foreign_worker) # Has values 2 and 1 - want to replace with 1 and 0
train$foreign_worker = ifelse(train$foreign_worker == 2, 0L, 1L) # Change 2 (no) to 0, integers
valid$foreign_worker = ifelse(valid$foreign_worker == 2, 0L, 1L) # Do the same for validation dataset
test$foreign_worker = ifelse(test$foreign_worker == 2, 0L, 1L) # Do the same for test dataset

library(dplyr)
# Change categorical variables (that aren't already encoded 1 or 0) to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing") # Leave out credit risk and foreign worker because they're already 1/0
train = train %>% mutate (across(all_of(cat_vars), as.factor))
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
test = test %>% mutate (across(all_of(cat_vars), as.factor))

# Note: for this algorithm, we do not need to standardize continuous variables

# Create dummy variables using model.matrix
train.d = model.matrix(credit_risk ~ ., data = train)[, -1]
valid.d = model.matrix(credit_risk ~ ., data = valid)[, -1]
test.d = model.matrix(credit_risk ~ ., data = test)[, -1]

# Turn the train/validation/test datasets into a particular format compatible with xgboost() function
library(xgboost)
dtrain <- xgb.DMatrix(data = train.d, label = train$credit_risk)
dvalid <- xgb.DMatrix(data = valid.d, label = valid$credit_risk)
dtest <- xgb.DMatrix(data = test.d, label = test$credit_risk)

# Create list of parameters to pass into xgboost() function
param_list = list(
  max.depth=20, 
  eta = 0.1,
  objective = "binary:logistic",
  min_child_weight=20,
  lambda = 1,
  alpha = 0.75
)

bst = xgboost(data = dtrain, 
              params = param_list,
              nrounds=200,
              verbose = 0)

library(pROC)
# Check AUC on training data to ensure overfitting isn't happening
train.pred = predict(bst, dtrain)
roc.train = roc(train$credit_risk, as.vector(train.pred))
auc(roc.train)

# Find AUC on validation data
valid.pred = predict(bst, dvalid)
roc.valid = roc(valid$credit_risk, as.vector(valid.pred))
auc(roc.valid)

# Identify optimal cutoff for classification based on ROC
opt_cutoff = coords(roc.valid, "best", ret = "threshold")
valid.class = ifelse(valid.pred > opt_cutoff[[1]], 1, 0)
library(caret)

# Confusion matrix on validation data
confusionMatrix(table(valid.class,valid$credit_risk), positive="1")

#Check AUC on test data
test.pred = predict(bst, dtest)
roc.test = roc(test$credit_risk, as.vector(test.pred))
auc(roc.test)

# Identify optimal cutoff for classification based on ROC
opt_cutoff = coords(roc.test, "best", ret = "threshold")
test.class = ifelse(test.pred > opt_cutoff[[1]], 1, 0)
library(caret)

# Confusion matrix on test data
cm = confusionMatrix(table(test.class,test$credit_risk), positive="1")

# Calculating F1 score
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1

# Write validation/test predictions to CSV files to later use as inputs for stacked classifier
write.csv(valid.pred, "XGB_pred_valid.csv")
write.csv(test.pred, "XGB_pred_test.csv")