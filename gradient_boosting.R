#Gradient boosting on filtered data

train = read.csv("train_fs_ig.csv")
valid = read.csv("valid_fs_ig.csv")
test = read.csv("test_fs_ig.csv")

colnames(train)

#convert categorical variables to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker")
#Note: we leave credit_risk as numeric because that's required

library(dplyr)
train = train %>% mutate (across(all_of(cat_vars), as.factor))

library(gbm)
library(MASS)

#optimal number of iterations/trees
gbm.model = gbm(
  formula = credit_risk ~ .,
  data = train,
  distribution = "bernoulli", #loss function for classification
  n.trees = 500, #max number of iterations, we will later find best one with gbm.perf
  interaction.depth = 1, 
  shrinkage = 0.1,
  cv.folds = 5,
  n.minobsinnode = 20, #minimum leaf size
  class.stratify.cv = TRUE #stratify cross validation by credit_risk
)
#Tune model
n_iter = gbm.perf(gbm.model, plot.it=FALSE, oobag.curve = FALSE, method = "cv")
#Try again
gbm.model = gbm(
  formula = credit_risk ~ .,
  data = train,
  distribution = "bernoulli", #loss function for classification
  n.trees = n_iter,
  interaction.depth = 1, 
  shrinkage = 0.1,
  cv.folds = 5,
  n.minobsinnode = 20, #minimum leaf size
  class.stratify.cv = TRUE #stratify cross validation by credit_risk
)

#Check training data AUC to verify overfitting is not occurring
train.pred= predict(gbm.model, n.trees = gbm.model$n.trees,
                    type = "response")
roc.train = roc(train$credit_risk, train.pred)
auc(roc.train)


#AUC on validation data
#Convert categorical variables in validation data to factors
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))


#Find AUC on validation data
gb_valid.pred= predict(gbm.model, newdata=valid, n.trees = gbm.model$n.trees,
                    type = "response") #get back probability instead of logit
library(pROC)
roc.valid = roc(valid$credit_risk, gb_valid.pred)
auc(roc.valid) 


#Find AUC on test data
#First, convert categorical vars to factors
test = test %>% mutate (across(all_of(cat_vars), as.factor))
gb_test.pred= predict(gbm.model, newdata=test, n.trees = gbm.model$n.trees,
                       type = "response") #get back probability instead of logit
library(pROC)
roc.test = roc(test$credit_risk, gb_test.pred)
auc(roc.test) 

#get optimal cutoff for event based on ROC
opt_cutoff = coords(roc.test, "best", ret = "threshold")

library(caret)
test.class = ifelse(gb_test.pred > opt_cutoff[[1]], 1, 0) #classify event based on cutoff
cm=confusionMatrix(table(test.class,test$credit_risk), positive="1") 
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1

write.csv(gb_valid.pred, "GB_pred_valid.csv")
write.csv(gb_test.pred, "GB_pred_test.csv")
