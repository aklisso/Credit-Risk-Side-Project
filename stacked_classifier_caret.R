train_df <- read.csv("train_fs_ig.csv")
valid_df <- read.csv("valid_fs_ig.csv")
test_df <- read.csv("test_fs_ig.csv")
library(dplyr)
#want to maintain same training set for consistency, but don't need validation anymore
test_df <- rbind(valid_df, test_df) 

#Change categorical variables to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker", "credit_risk")
train_df = train_df %>% mutate (across(all_of(cat_vars), as.factor))
train_df$duration = as.numeric(train_df$duration) #ensure continuous
train_df = train_df %>% mutate (log.duration = log(duration))#log transform to reduce skewness
train_df = train_df %>% select (-duration) #get rid of original duration

test_df = test_df %>% mutate (across(all_of(cat_vars), as.factor))
test_df$duration = as.numeric(test_df$duration) #ensure continuous
test_df = test_df %>% mutate (log.duration = log(duration))#log transform to reduce skewness
test_df = test_df %>% select (-duration) #get rid of original duration


#Rename credit risk values to "good" (0) and "bad" (1) because R will throw a fit later if I don't
train_df$credit_risk = if_else(train_df$credit_risk==1, "bad_credit_risk", "good_credit_risk")
test_df$credit_risk = if_else(test_df$credit_risk==1, "bad_credit_risk", "good_credit_risk")

#Convert data to dummy variables
library(caret)
dummies <- dummyVars(credit_risk~., data = train_df)
train_dummies = predict(dummies, newdata = train_df)
train_df = data.frame (credit_risk = train_df$credit_risk, train_dummies)
#Apply same dummy variables from training data to test data for consistency
test_dummies = predict(dummies, newdata = test_df)
test_df = data.frame(credit_risk = test_df$credit_risk, test_dummies)

#Center/scale continuous variables based on training data
preproc = preProcess (train_df %>%
                       dplyr::select(log.duration),
                     method = c("center", "scale"))

#apply preprocessing to training AND testing data
train_proc=predict(preproc, newdata = train_df %>% select (log.duration))
train_df = cbind (train_proc, train_df %>% select(-log.duration)) #note: log.duration is now standardized
train_df = train_df %>% rename (log.duration.s = log.duration)

test_proc=predict(preproc, newdata = test_df %>% select (log.duration))
test_df = cbind (test_proc, test_df %>% select(-log.duration))
test_df = test_df %>% rename (log.duration.s = log.duration)

ggplot(data = train_df, aes(x=log.duration.s)) + geom_histogram() #looks standardized/symmetric, mean of 0. good!

library(caretEnsemble)

train_control = trainControl(
  method = "cv",
  number = 5,#5 folds
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

model_list = caretEnsemble::caretList(
  credit_risk ~.,
  data = train_df,
  methodList = c("rf", "gbm", "knn", "nnet", "glmnet", "xgbTree"), 
  metric = "ROC", 
  trControl = train_control
)

modelCor(resamples(model_list)) #ensure models are not redundant/too similar, otherwise won't add value to stacked classifier
#Random forest has high correlation with glmnet (>0.8). Glmnet and nnet also have moderate to high correlation (0.79)
#Glmnet is not adding that much information - might be helpful to remove it.

#model list with Glmnet removed
model_list2 = caretEnsemble::caretList(
  credit_risk ~.,
  data = train_df,
  methodList = c("rf", "gbm", "knn", "nnet", "xgbTree"), 
  metric = "ROC", 
  trControl = train_control
)

#Generate predicted values for each of the base classifiers
model_preds = predict(model_list2, newdata = test_df, excluded_class_id = 2L)

#AUC of individual predictions
indiv_aucs = caTools::colAUC(model_preds, test_df$credit_risk)

#Generate predicted values for stacked model
stacked_model = caretEnsemble::caretStack(model_list2, method = "glm", metric = "ROC")

#Add the stacked model's predictions to the predictions dataframe
model_preds$stacked = predict(stacked_model, newdata = test_df, excluded_class_id = 2L)
all_aucs = caTools::colAUC(model_preds, test_df$credit_risk)
#Stacked classifier has AUC of 0.7793865 on testing data - in fact, xgbTree slightly outperforms it.

#Get predicted probabilities as vector
stacked_preds <- as.numeric(predict(stacked_model, newdata = test_df, excluded_class_id = 2L)[['bad_credit_risk']])
test_outcomes = if_else(test_df$credit_risk == "bad_credit_risk", 1, 0)
test_roc = roc(test_outcomes, stacked_preds)
auc(test_roc)
opt_cutoff = coords(test_roc, "best", ret = "threshold")
#Get predicted classes (0,1)
test_class = ifelse(stacked_preds > opt_cutoff[[1]], 1, 0)
#Confusion matrix
cm = confusionMatrix(table(test_class,test_outcomes), positive="1") 
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1




#Knn has a super low AUC (almost 0.5) - let's see if removing it helps.

#model list without KNN
model_list3 = caretEnsemble::caretList(
  credit_risk ~.,
  data = train_df,
  methodList = c("rf", "gbm", "nnet", "xgbTree"), 
  metric = "ROC", 
  trControl = train_control
)

#Generate predicted values for each of the base classifiers
model_preds = predict(model_list3, newdata = test_df, excluded_class_id = 2L)

#AUC of individual predictions
indiv_aucs = caTools::colAUC(model_preds, test_df$credit_risk)

#Generate predicted values for stacked model
stacked_model = caretEnsemble::caretStack(model_list3, method = "glm", metric = "ROC")

#Add the stacked model's predictions to the predictions dataframe
model_preds$stacked = predict(stacked_model, newdata = test_df, excluded_class_id = 2L)
all_aucs = caTools::colAUC(model_preds, test_df$credit_risk)
#AUC is 0.7764179 - slightly lower, so let's leave KNN in.
