#Random forest on filtered data

#Read in the feature-selected data
train = read.csv("train_fs_ig.csv")
#Convert categorical variables to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration", 
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker","credit_risk")
library(dplyr)
train = train %>% mutate (across(all_of(cat_vars), as.factor))

library(randomForest) #most well-known implementation of random forest in R

set.seed(2025)
rf.model = randomForest(
  formula = credit_risk~.,
  data = train
)

#Check training data AUC to verify overfitting is not occurring
train.pred= predict(rf.model, type = "prob")
train.pred = train.pred[,2]
roc.train = roc(train$credit_risk, train.pred)
auc(roc.train) #AUC is 0.79 - doesn't look like overfitting is happening :) 


#AUC on validation data
valid = read.csv("valid_fs_ig.csv")
#Convert categorical variables in validation data to factors
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
#Find AUC on validation data
valid.pred= predict(rf.model, newdata=valid, type = "prob")
valid.pred = as.vector(valid.pred[,2])

library(pROC)
roc.valid = roc(valid$credit_risk, valid.pred)
auc(roc.valid) 


#get optimal cutoff for event based on ROC
opt_cutoff = coords(roc.valid, "best", ret = "threshold")


library(caret)
valid.class = ifelse(valid.pred > opt_cutoff[[1]], 1, 0) #classify event based on cutoff
confusionMatrix(table(valid.class,valid$credit_risk), positive="1") #get confusion matrix values

write.csv(valid.pred, "RF_pred_valid.csv", row.names=FALSE)


