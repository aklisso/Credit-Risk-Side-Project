#Create meta-model which learns from predictions of base models

#Read in base models' predictions on validation set
#note: this code isn't going to work unless we re-run the code to generate the CSV's below, since I changed the 
# train-test split and have not re-run the code yet.

#Read in predictions on validation data
knn_valid = as.vector(read.csv("KNN_pred_valid.csv")[[2]]) #K nearest neighbors
rf_valid = as.vector(read.csv("RF_pred_valid.csv")[[2]]) #Random forest
encv_valid = as.vector(read.csv("ENCV_pred_valid.csv")[[2]]) #Elastic net (cross validated for optimal lambda)
gb_valid = as.vector(read.csv("GB_pred_valid.csv")[[2]]) #Gradient boosting
ann_valid = as.vector(read.csv("ANN_pred_valid.csv")[[2]]) #Artificial neural network
xgb_valid = as.vector(read.csv("XGB_pred_valid.csv")[[2]]) #Extreme Gradient Boosting
credit_risk_valid = read.csv("valid_fs_ig.csv")$credit_risk #actual target variable in validation set


#Create dataframe with each base model's predicted probabilities of poor credit risk on validation set
valid_probs = data.frame(
  knn = knn_valid,
  rf = rf_valid,
  encv = encv_valid,
  gb = gb_valid,
  ann = ann_valid,
  xgb = xgb_valid,
  credit_risk = credit_risk_valid
)

View(valid_probs)

library(dplyr)
valid_probs = valid_probs %>% mutate (credit_risk = as.factor(credit_risk))

#meta model: linear regression
model <- glm(credit_risk~., data= valid_probs, family = binomial(link="logit")) #still need to evaluate

valid_pred_probs = as.vector(predict(model, valid_probs, type = "response")) #predicted probabilities

library(pROC)
model.roc = roc(response = valid_probs$credit_risk, 
                predictor = valid_pred_probs)
auc(model.roc) 

#now try on testing data
#Read in predictions on validation data
knn_test = as.vector(read.csv("KNN_pred_test.csv")[[2]]) #K nearest neighbors
rf_test = as.vector(read.csv("RF_pred_test.csv")[[2]]) #Random forest
encv_test = as.vector(read.csv("ENCV_pred_test.csv")[[2]]) #Elastic net (cross validated for optimal lambda)
gb_test = as.vector(read.csv("GB_pred_test.csv")[[2]]) #Gradient boosting
ann_test = as.vector(read.csv("ANN_pred_test.csv")[[2]]) #Artificial neural network
xgb_test = as.vector(read.csv("XGB_pred_test.csv")[[2]]) #Extreme Gradient Boosting
credit_risk_test = read.csv("test_fs_ig.csv")$credit_risk #actual target variable in validation set


#Create dataframe with each base model's predicted probabilities of poor credit risk on validation set
test_probs = data.frame(
  knn = knn_test,
  rf = rf_test,
  encv = encv_test,
  gb = gb_test,
  ann = ann_test,
  xgb = xgb_test,
  credit_risk = credit_risk_test
)

#Calculate predicted probabilities using meta model
test_pred_probs = as.vector(predict(model, test_probs, type = "response")) #predicted probabilities

#Calculate AUC
library(pROC)
model.roc = roc(response = test_probs$credit_risk, 
                predictor = test_pred_probs)
auc(model.roc) 

#Get optimal cutoff for events
opt_cutoff = coords(model.roc, "best", ret = "threshold")

library(caret)
test.class = ifelse(test_pred_probs > opt_cutoff[[1]], 1, 0) #classify event based on cutoff
cm=confusionMatrix(table(test.class,test_probs$credit_risk), positive="1") 
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1


#----------------------------------------------------------------------------------------------------------------------------------------
#try taking weighted avg of validation predictions
valid_probs$avg = rowMeans(valid_probs[,-7])
roc.valid.mean = roc(valid_probs$credit_risk, valid_probs$avg)
auc(roc.valid.mean) #not quite as good- stacking is the way to go!