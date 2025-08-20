#Read in CSV's created in cleaning_eda.R
train <- read.csv("credit_train.csv")
valid <- read.csv("credit_valid.csv")
test <- read.csv("credit_test.csv")

#Standardize numeric variables: duration, amount, age

#Train
library(dplyr)
train.s = train %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
train.s = train.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
train.s = train.s %>% mutate (age.s = (age-mean(age))/sd(age))
train.s = train.s %>% select(-c(duration, amount, age))

#Validation
valid.s = valid %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
valid.s = valid.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
valid.s = valid.s %>% mutate (age.s = (age-mean(age))/sd(age))
valid.s = valid.s %>% select(-c(duration, amount, age))

#Test
test.s = test %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
test.s = test.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
test.s = test.s %>% mutate (age.s = (age-mean(age))/sd(age))
test.s = test.s %>% select(-c(duration, amount, age))

#elasticnet regression with cross validation on training data
library(glmnet)
set.seed(2025)
X = model.matrix(credit_risk ~ ., data= train.s) [,-1]
y = train.s$credit_risk

#Find best lambda using CV
credit_elasticnet = cv.glmnet(x=X, y=y, alpha=1, family=binomial(link="logit"))
cv_elasticnet_lambda = credit_elasticnet$lambda.1se #best MSE achieved when lambda = 0.01738841

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
auc(cv_elastic_roc) #AUC is 0.7486 - not awful, not amazing
#Try to find cutoff for event that maximizes sensitivity/specificity
opt_cutoff = coords(cv_elastic_roc, "best", ret = "threshold")
#More stringent threshold than 0.5 for events (~0.6150302)

#Get predicted classes (0,1) for validation data
en_v_class = ifelse(en_v_pred > 0.6150302, 1, 0)

#Confusion matrix
library(caret)
confusionMatrix(table(en_v_class,valid$credit_risk))
#Everything looks good to me except PPV (probability someone pred. risky truly is risky) is a bit low
#This means the bank could miss out on $$$ from loan interest if they don't approve the loan

#Next: conduct research to see what the optimal balance of sens/spec is in this scenario and
# adjust cutoff accordingly



