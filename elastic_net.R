# Using elastic net regression to predict binary outcome: credit risk (good vs poor)

# Read in the data (cleaned + feature selection implemented)
train = read.csv("train_fs_ig.csv")
valid = read.csv("valid_fs_ig.csv")
test = read.csv("test_fs_ig.csv")

# Standardize continuous variables: duration, amount
library(dplyr)

# Training data
train.s = train %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
train.s = train.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
train.s = train.s %>% dplyr::select(-c(duration, amount))

# Validation data
valid.s = valid %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
valid.s = valid.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
valid.s = valid.s %>% dplyr::select(-c(duration, amount))

# Test data
test.s = test %>% mutate (duration.s = (duration-mean(duration))/sd(duration))
test.s = test.s %>% mutate (amount.s = (amount-mean(amount))/sd(amount))
test.s = test.s %>% dplyr::select(-c(duration, amount))


# Factorize categorical variables
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker", "credit_risk")
train = train %>% mutate (across(all_of(cat_vars), as.factor))
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
test = test %>% mutate (across(all_of(cat_vars), as.factor))

# Peform elasticnet regression with cross validation on training data
library(glmnet)
set.seed(2025)
X = model.matrix(credit_risk ~ ., data= train.s) [,-1]
y = train.s$credit_risk

#Find the best lambda using cross validation
credit_elasticnet = cv.glmnet(x=X, y=y, alpha=1, family=binomial(link="logit"))
cv_elasticnet_lambda = credit_elasticnet$lambda.1se 

# Coefficients for elasticnet
coef(credit_elasticnet, s = cv_elasticnet_lambda)

# Predicted values for elasticnet (ensure they look reasonable)
train.pred = predict(credit_elasticnet, X, s = cv_elasticnet_lambda, type = "response")
# Note: type = "response" ensures we get predicted probabilities (what we want). The default is "link" which gives logits


library(pROC)
set.seed(2025)
# Get validation data into proper format for prediction
Xv = model.matrix(credit_risk ~ ., data= valid.s) [,-1]
yv = valid.s$credit_risk
# Get predictions on validation data (must convert to vector because predict() returns a matrix
valid.pred = as.vector(predict(credit_elasticnet, Xv, s = cv_elasticnet_lambda, type= "response"))
# Find AUC on validation data
en_v_roc = roc(valid$credit_risk, valid.pred)
auc(en_v_roc) 
# Identify optimal classification cutoff using ROC
opt_cutoff = coords(en_v_roc, "best", ret = "threshold")
# Get predicted classes (0,1) for validation data
valid.class = ifelse(valid.pred > opt_cutoff[[1]], 1, 0)
# Confusion matrix on validation data
library(caret)
confusionMatrix(table(valid.class,valid$credit_risk), positive="1") 

# Get predictions on validation data
set.seed(2025)
# Get test data into proper format for prediction
Xtest = model.matrix(credit_risk ~ ., data= test.s) [,-1]
ytest = test.s$credit_risk
# Get predictions on test data (must convert to vector because predict() returns a matrix
test.pred = as.vector(predict(credit_elasticnet, Xtest, s = cv_elasticnet_lambda, type= "response"))
en_test_roc = roc(test$credit_risk, test.pred)
auc(en_test_roc) 
# Identify optimal classification cutoff using ROC
opt_cutoff = coords(en_test_roc, "best", ret = "threshold")
# Get predicted classes (0,1) for test data
test.class = ifelse(test.pred > opt_cutoff[[1]], 1, 0)
# Confusion matrix on test data
library(caret)
cm = confusionMatrix(table(test.class,test$credit_risk), positive="1") 

# Calculate F1 score 
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1

# Write validation/test predictions to CSV files to later use as inputs for stacked classifier
write.csv(valid.pred, "ENCV_pred_valid.csv")
write.csv(test.pred, "ENCV_pred_test.csv")

