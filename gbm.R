# Using gradient boosting method (GBM) to predict binary outcome: credit risk (good vs poor)

# Read in the data (cleaned + feature selection implemented)
train = read.csv("train_fs_ig.csv")
valid = read.csv("valid_fs_ig.csv")
test = read.csv("test_fs_ig.csv")

# Change categorical variables to factors for the gbm() function
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker")
# Note: credit_risk is left as a numeric 0/1 variable as required by the function
library(dplyr)
train = train %>% mutate (across(all_of(cat_vars), as.factor))

library(gbm)
library(MASS)

# Build an initial GBM model using cross-validation to find the best number of trees
gbm.model = gbm(
  formula = credit_risk ~ .,
  data = train,
  distribution = "bernoulli", 
  n.trees = 500, # 
  interaction.depth = 1, 
  shrinkage = 0.1, 
  cv.folds = 5, 
  n.minobsinnode = 20, 
  class.stratify.cv = TRUE 
)

# Identify the optimal number of iterations
n_iter = gbm.perf(gbm.model, plot.it=FALSE, oobag.curve = FALSE, method = "cv")

# Retrain the final model using the optimal number of trees
gbm.model = gbm(
  formula = credit_risk ~ .,
  data = train,
  distribution = "bernoulli",
  n.trees = n_iter, 
  interaction.depth = 1, 
  shrinkage = 0.1,
  cv.folds = 5,
  n.minobsinnode = 20,
  class.stratify.cv = TRUE
)

# Check AUC on training data to ensure overfitting isn't happening
train.pred = predict(gbm.model, n.trees = gbm.model$n.trees,
                     type = "response")
library(pROC)
roc.train = roc(train$credit_risk, train.pred)
auc(roc.train)


# Convert categorical variables to factors in validation data
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))


# Generate predictions/calculate AUC on the validation data
valid.pred = predict(gbm.model, newdata=valid, n.trees = gbm.model$n.trees,
                        type = "response") 
roc.valid = roc(valid$credit_risk, valid.pred)
auc(roc.valid) 


# Convert categorical variables to factors in test data
test = test %>% mutate (across(all_of(cat_vars), as.factor))


# Generate predictions and calculate AUC on the test data
test.pred = predict(gbm.model, newdata=test, n.trees = gbm.model$n.trees,
                       type = "response")
roc.test = roc(test$credit_risk, test.pred)
auc(roc.test) 

# Identify optimal classification cutoff using ROC
opt_cutoff = coords(roc.test, "best", ret = "threshold")

# Confusion matrix on test data
library(caret)
test.class = ifelse(test.pred > opt_cutoff[[1]], 1, 0) 
cm=confusionMatrix(table(test.class,test$credit_risk), positive="1") 

# Calculate F1 score
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1

# Write validation/test predictions to CSV files to later use as inputs for stacked classifier
write.csv(valid.pred, "GB_pred_valid.csv")
write.csv(test.pred, "GB_pred_test.csv")