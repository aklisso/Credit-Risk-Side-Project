#XGBoost
train = read.csv("train_fs_ig.csv")
valid = read.csv("valid_fs_ig.csv")
test = read.csv("test_fs_ig.csv")


#Ensure all binary variables have levels 0 and 1 (not 2 and 1)
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

#Note: unlike ANN, we do not need to standardize quantitative variables

#create dummy variables using model.matrix
#Note: I ran unique(train$var) and unique(valid$var) for each categorical variable to ensure that none had levels the other lacked, which would mess
# up the dummy encoding of the variables
train.d = model.matrix(credit_risk ~ ., data = train)[, -1]
valid.d = model.matrix(credit_risk ~ ., data = valid)[, -1]
test.d = model.matrix(credit_risk ~ ., data = test)[, -1]

library(xgboost)
dtrain <- xgb.DMatrix(data = train.d, label = train$credit_risk)
dvalid <- xgb.DMatrix(data = valid.d, label = valid$credit_risk)
dtest <- xgb.DMatrix(data = test.d, label = test$credit_risk)

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
#Check AUC on training data to ensure overfitting isn't happening
xgb_train_pred = predict(bst, dtrain)
roc.train = roc(train$credit_risk, as.vector(xgb_train_pred))
auc(roc.train)

#Check AUC on validation data
xgb_valid_pred = predict(bst, dvalid)
roc.valid = roc(valid$credit_risk, as.vector(xgb_valid_pred))
auc(roc.valid)
#confusion matrix on validation data
opt_cutoff = coords(roc.valid, "best", ret = "threshold")
xgb_valid_class = ifelse(xgb_valid_pred > opt_cutoff[[1]], 1, 0)
library(caret)
confusionMatrix(table(xgb_valid_class,valid$credit_risk), positive="1")

#Check AUC on test data
xgb_test_pred = predict(bst, dtest)
roc.test = roc(test$credit_risk, as.vector(xgb_test_pred))
auc(roc.test)
#confusion matrix on validation data
opt_cutoff = coords(roc.test, "best", ret = "threshold")
xgb_test_class = ifelse(xgb_test_pred > opt_cutoff[[1]], 1, 0)
library(caret)
cm = confusionMatrix(table(xgb_test_class,test$credit_risk), positive="1")
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1

write.csv(xgb_valid_pred, "XGB_pred_valid.csv")
write.csv(xgb_test_pred, "XGB_pred_test.csv")