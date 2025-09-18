#Read in CSV's created in cleaning_eda.R
train <- read.csv("train_fs_ig.csv")
valid <- read.csv("valid_fs_ig.csv")
test <- read.csv("test_fs_ig.csv")

#Standardize numeric variables: duration, amount (note: age removed in feature seln earlier)
library(dplyr)
train.s = train %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
train.s = train.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
train.s = train.s %>% dplyr::select(-c(duration, amount))

#Validation
valid.s = valid %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
valid.s = valid.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
valid.s = valid.s %>% dplyr::select(-c(duration, amount))

#Test
test.s = test %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
test.s = test.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
test.s = test.s %>% dplyr::select(-c(duration, amount))


#Factorize categorical variables
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker", "credit_risk")
train = train %>% mutate (across(all_of(cat_vars), as.factor))
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
test = test %>% mutate (across(all_of(cat_vars), as.factor))

#elasticnet regression with cross validation on training data
library(glmnet)
set.seed(2025)
X = model.matrix(credit_risk ~ ., data= train.s) [,-1]
y = train.s$credit_risk

#Find best lambda using CV
credit_elasticnet = cv.glmnet(x=X, y=y, alpha=1, family=binomial(link="logit"))
cv_elasticnet_lambda = credit_elasticnet$lambda.1se 

#Coefficients for elasticnet
coef(credit_elasticnet, s = cv_elasticnet_lambda)

#Predicted values for elasticnet (ensure they look reasonable)
en_train_pred = predict(credit_elasticnet, X, s = cv_elasticnet_lambda, type = "response")
# note: type = "response" gives predicted probabilities. default is "link" which gives logits

#Find AUC on validation data
library(pROC)
#Get predictions on validation data
set.seed(2025)
Xv = model.matrix(credit_risk ~ ., data= valid.s) [,-1]
yv = valid.s$credit_risk
#Must convert to vector because predict() returns a matrix
en_v_pred = as.vector(predict(credit_elasticnet, Xv, s = cv_elasticnet_lambda, type= "response"))
cv_elastic_roc = roc(valid$credit_risk, en_v_pred)
auc(cv_elastic_roc) 
#Try to find cutoff for event that maximizes sensitivity/specificity
opt_cutoff = coords(cv_elastic_roc, "best", ret = "threshold")
#Get predicted classes (0,1) for validation data
en_v_class = ifelse(en_v_pred > opt_cutoff[[1]], 1, 0)
#Confusion matrix
library(caret)
confusionMatrix(table(en_v_class,valid$credit_risk), positive="1") 


#Find AUC on test data
library(pROC)
#Get predictions on validation data
set.seed(2025)
Xtest = model.matrix(credit_risk ~ ., data= test.s) [,-1]
ytest = test.s$credit_risk
#Must convert to vector because predict() returns a matrix
en_test_pred = as.vector(predict(credit_elasticnet, Xtest, s = cv_elasticnet_lambda, type= "response"))
test_elastic_roc = roc(test$credit_risk, en_test_pred)
auc(test_elastic_roc) 
#Try to find cutoff for event that maximizes sensitivity/specificity
opt_cutoff = coords(test_elastic_roc, "best", ret = "threshold")
#Get predicted classes (0,1) for validation data
en_test_class = ifelse(en_test_pred > opt_cutoff[[1]], 1, 0)
#Confusion matrix
library(caret)
cm = confusionMatrix(table(en_test_class,test$credit_risk), positive="1") 
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1

write.csv(en_v_pred, "ENCV_pred_valid.csv")
write.csv(en_test_pred, "ENCV_pred_test.csv")

