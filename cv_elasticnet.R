#Read in CSV's created in cleaning_eda.R
train <- read.csv("train_fs_ig.csv")
valid <- read.csv("valid_fs_ig.csv")
test <- read.csv("test_fs_ig.csv")

#Standardize numeric variables: duration, amount (note: age removed in feature seln earlier)
library(dplyr)
train.s = train %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
train.s = train.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
train.s = train.s %>% select(-c(duration, amount))

#Validation
valid.s = valid %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
valid.s = valid.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
valid.s = valid.s %>% select(-c(duration, amount))

#Test
test.s = test %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
test.s = test.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
test.s = test.s %>% select(-c(duration, amount))

#elasticnet regression with cross validation on training data
library(glmnet)
set.seed(2025)
X = model.matrix(credit_risk ~ ., data= train.s) [,-1]
y = train.s$credit_risk

#Find best lambda using CV
credit_elasticnet = cv.glmnet(x=X, y=y, alpha=1, family=binomial(link="logit"))
cv_elasticnet_lambda = credit_elasticnet$lambda.1se #best MSE achieved when lambda = 0.02936188

#Coefficients for elasticnet
coef(credit_elasticnet, s = cv_elasticnet_lambda)

#Predicted values for elasticnet
en_t_pred = predict(credit_elasticnet, X, s = cv_elasticnet_lambda, type = "response")
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
auc(cv_elastic_roc) #AUC is 0.7589
#Try to find cutoff for event that maximizes sensitivity/specificity: 0.6443343
opt_cutoff = coords(cv_elastic_roc, "best", ret = "threshold")

#Get predicted classes (0,1) for validation data
en_v_class = ifelse(en_v_pred > opt_cutoff[[1]], 1, 0)

#Confusion matrix
library(caret)
confusionMatrix(table(en_v_class,valid$credit_risk), positive="1")




