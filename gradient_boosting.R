#Gradient boosting on filtered data

train = read.csv("train_fs_ig.csv")

#convert categorical variables to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration", "installment_rate", 
             "personal_status_sex", "other_debtors", "present_residence", 
             "property", "other_installment_plans","housing",
             "job","foreign_worker")
#Note: we leave credit_risk as numeric because that's required for gbm

library(dplyr)
train = train %>% mutate (across(all_of(cat_vars), as.factor))

library(gbm)
library(MASS)

#optimal number of iterations/trees
n_iter = gbm.perf(gbm.model, plot.it=TRUE, oobag.curve = FALSE, method = "cv")

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



#AUC on validation data
valid = read.csv("valid_fs_ig.csv")
#Convert categorical variables in validation data to factors
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
#Find AUC on validation data
valid.pred= predict(gbm.model, newdata=valid, n.trees = gbm.model$n.trees,
                    type = "response") #get back probability instead of logit
library(pROC)
roc.valid = roc(valid$credit_risk, valid.pred)
auc(roc.valid) 


#Check training data AUC to verify overfitting is not occurring
train.pred= predict(gbm.model, n.trees = gbm.model$n.trees,
                    type = "response")
roc.train = roc(train$credit_risk, train.pred)
auc(roc.train)


#is AUC on training = 0.833 and AUC on validation 0.778 ok?

