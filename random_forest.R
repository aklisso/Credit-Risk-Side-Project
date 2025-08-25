#Random forest on filtered data

#Read in the feature-selected data
train = read.csv("train_fs_ig.csv")

#Convert categorical variables to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration", "installment_rate", 
             "personal_status_sex", "other_debtors", "present_residence", 
             "property", "other_installment_plans","housing",
             "job","foreign_worker","credit_risk")
library(dplyr)
train = train %>% mutate (across(all_of(cat_vars), as.factor))

library(randomForest) #most well-known implementation of random forest in R

set.seed(2025)
rf.model = randomForest(
  formula = credit_risk~.,
  data = train
)

tail(rf.model$err.rate) #error looks weirdly high for nonevents

#AUC on validation data
valid = read.csv("valid_fs_ig.csv")
#Convert categorical variables in validation data to factors
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
#Find AUC on validation data
valid.pred= predict(rf.model, newdata=valid, type = "prob")
valid.pred = valid.pred[,2]
library(pROC)
roc.valid = roc(valid$credit_risk, valid.pred)
auc(roc.valid) #not horrible- 0.778


#Check training data AUC to verify overfitting is not occurring
train.pred= predict(rf.model, type = "prob")
train.pred = train.pred[,2]
roc.train = roc(train$credit_risk, train.pred)
auc(roc.train) #AUC is 0.79 - doesn't look like overfitting is happening :) 

#Next- tune hyperparameters



