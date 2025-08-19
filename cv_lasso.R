#Read in CSV's created in cleaning_eda.R
train <- read.csv("credit_train.csv")
valid <- read.csv("credit_valid.csv")
test <- read.csv("credit_test.csv")

#Standardize numeric variables: duration, amount, age

#Train
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

#Lasso regression with cross validation on training data
library(glmnet)
set.seed(2025)
X = model.matrix(credit_risk ~ ., data= train.s) [,-1]
y = train.s$credit_risk

#Find best lambda using CV
credit_lasso = cv.glmnet(x=X, y=y, alpha=1, family=binomial(link="logit"))
cv_lasso_lambda = credit_lasso$lambda.1se #best MSE achieved when lambda = 0.01738841

#Coefficients for Lasso
coef(credit_lasso, s = cv_lasso_lambda)

#Predicted values for Lasso
lasso_pred = predict(credit_lasso, X, s = cv_lasso_lambda, type = "response")
# note: type = "response" gives predicted probabilities. default is "link" which gives logits