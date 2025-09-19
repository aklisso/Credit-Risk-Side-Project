# Create stacked classifier using caretStack from caret package

# Read in the data
train_df = read.csv("train_fs_ig.csv")
valid_df = read.csv("valid_fs_ig.csv")
test_df = read.csv("test_fs_ig.csv")
library(dplyr)

# We want to maintain the same training dataset for consistency, but we don't need a validation dataset anymore since caret uses CV
test_df <- rbind(valid_df, test_df) 

# Change categorical variables to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker", "credit_risk")
train_df = train_df %>% mutate (across(all_of(cat_vars), as.factor))
 
# Address skewness in continuous variables
train_df$duration = as.numeric(train_df$duration) # Ensure duration is continuous
train_df = train_df %>% mutate (log.duration = log(duration))# Log transform duration to reduce skewness
train_df = train_df %>% select (-duration) # Get rid of original duration variable
train_df$amount = as.numeric(train_df$amount) # Ensure amount is continuous
train_df = train_df %>% mutate (log.amount = log(amount))# Log transform amount to reduce skewness
train_df = train_df %>% select (-amount) # Get rid of original amount variable

# Repeat for test data
test_df = test_df %>% mutate (across(all_of(cat_vars), as.factor))
test_df$duration = as.numeric(test_df$duration)
test_df = test_df %>% mutate (log.duration = log(duration))
test_df = test_df %>% select (-duration)
test_df$amount = as.numeric(test_df$amount)
test_df = test_df %>% mutate (log.amount = log(amount))
test_df = test_df %>% select (-amount)

# Rename credit risk values to "good" (0) and "bad" (1) because caret likes them to be characters, not numbers
train_df$credit_risk = if_else(train_df$credit_risk==1, "bad_credit_risk", "good_credit_risk")
test_df$credit_risk = if_else(test_df$credit_risk==1, "bad_credit_risk", "good_credit_risk")

# Convert data to dummy variables
library(caret)
dummies <- dummyVars(credit_risk~., data = train_df)
train_dummies = predict(dummies, newdata = train_df)
train_df = data.frame (credit_risk = train_df$credit_risk, train_dummies)
# Apply same dummy variables from training data to test data for consistency
test_dummies = predict(dummies, newdata = test_df)
test_df = data.frame(credit_risk = test_df$credit_risk, test_dummies)

# Center/scale continuous variables based on training data
preproc = preProcess (train_df %>%
                       dplyr::select(log.duration, log.amount),
                     method = c("center", "scale"))

#apply preprocessing to training AND testing data
train_proc=predict(preproc, newdata = train_df %>% select (log.duration, log.amount))
train_df = cbind (train_proc, train_df %>% select(-log.duration, -log.amount))
train_df = train_df %>% rename (log.duration.s = log.duration, log.amount.s = log.amount)

test_proc=predict(preproc, newdata = test_df %>% select (log.duration, log.amount))
test_df = cbind (test_proc, test_df %>% select(-log.duration, -log.amount))
test_df = test_df %>% rename (log.duration.s = log.duration, log.amount.s = log.amount)

# Ensure standardization was performed correctly and skewness is no longer an issue
ggplot(data = train_df, aes(x=log.duration.s)) + geom_histogram()
ggplot(data = train_df, aes(x=log.amount.s)) + geom_histogram()

library(caretEnsemble)
# Create trainControl object to be passed into caretStack() function later
train_control = trainControl(
  method = "cv",
  number = 5,#5 folds
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Create list of models to be used in stacked classifier
model_list = caretEnsemble::caretList(
  credit_risk ~.,
  data = train_df,
  methodList = c("rf", "gbm", "knn", "nnet", "glmnet", "xgbTree"), 
  metric = "ROC", 
  trControl = train_control
)

# Look at correlation between models to ensure they are not too similar, otherwise they won't add 
# value to the stacked classifier
modelCor(resamples(model_list)) 
# GBM and KNN have high correlation
# RF and NNET have high correlation

# Let's see which ones to remove by examining AUC's.
# Generate predictions
model_preds = predict(model_list, newdata = test_df, excluded_class_id = 2L)
caTools::colAUC(model_preds, test_df$credit_risk)

# Between GBM and KNN, GBM has higher AUC - remove KNN.
# Between RF and NNET, NNET has higher AUC - remove RF.

# Rerun stacked classifier with redundant models removed
model_list2 = caretEnsemble::caretList(
  credit_risk ~.,
  data = train_df,
  methodList = c("gbm", "nnet", "glmnet", "xgbTree"), 
  metric = "ROC", 
  trControl = train_control
)

# Generate predicted values for stacked model
stacked_model = caretEnsemble::caretStack(model_list2, method = "glm", metric = "ROC")

# Add the stacked model's predictions to the preexisting dataframe of base models' predicted values
model_preds$stacked = predict(stacked_model, newdata = test_df, excluded_class_id = 2L)

# Compare AUC for all base models and stacked classifier
caTools::colAUC(model_preds, test_df$credit_risk)

#Stacked classifier has AUC of 0.7779282 on testing data - narrowly outperforms GBM and XGBtree.

# Get stacked model's predicted probabilities as vector
stacked_preds <- as.numeric(predict(stacked_model, newdata = test_df, excluded_class_id = 2L)[['bad_credit_risk']])
test_outcomes = if_else(test_df$credit_risk == "bad_credit_risk", 1, 0)
# Calculate AUC on testing data
test_roc = roc(test_outcomes, stacked_preds)
auc(test_roc)
# Determine optimal cutoff using ROC
opt_cutoff = coords(test_roc, "best", ret = "threshold")

# Classify events based on optimal cutoff
test_class = ifelse(stacked_preds > opt_cutoff[[1]], 1, 0)

# Confusion matrix
cm = confusionMatrix(table(test_class,test_outcomes), positive="1") 

# Calculate F1 score
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1