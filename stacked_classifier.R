# Create meta-model which learns from predictions of base models

# Read in base models' predictions on validation set
knn_valid = as.vector(read.csv("KNN_pred_valid.csv")[[2]]) # K nearest neighbors
rf_valid = as.vector(read.csv("RF_pred_valid.csv")[[2]]) # Random forest
encv_valid = as.vector(read.csv("ENCV_pred_valid.csv")[[2]]) # Elastic net (cross validated for optimal lambda)
gb_valid = as.vector(read.csv("GB_pred_valid.csv")[[2]]) # Gradient boosting
ann_valid = as.vector(read.csv("ANN_pred_valid.csv")[[2]]) # Artificial neural network
xgb_valid = as.vector(read.csv("XGB_pred_valid.csv")[[2]]) # Extreme Gradient Boosting
credit_risk_valid = read.csv("valid_fs_ig.csv")$credit_risk # Actual target variable in validation set

# Create dataframe with each base model's predicted probabilities of poor credit risk on validation set
valid_pred_df = data.frame(
  knn = knn_valid,
  rf = rf_valid,
  encv = encv_valid,
  gb = gb_valid,
  ann = ann_valid,
  xgb = xgb_valid,
  credit_risk = credit_risk_valid
)

# Convert credit_risk to factor
library(dplyr)
valid_pred_df = valid_pred_df %>% mutate (credit_risk = as.factor(credit_risk))

# Train meta model: logistic regression
model <- glm(credit_risk~., data= valid_pred_df, family = binomial(link="logit"))

# Predicted probabilities on validation data
valid.pred = as.vector(predict(model, valid_pred_df, type = "response")) 

library(pROC)
model.roc = roc(response = valid_pred_df$credit_risk, 
                predictor = valid.pred)
auc(model.roc) 

# Now try on testing data
# Read in base models' predictions on test data
knn_test = as.vector(read.csv("KNN_pred_test.csv")[[2]]) # K nearest neighbors
rf_test = as.vector(read.csv("RF_pred_test.csv")[[2]]) # Random forest
encv_test = as.vector(read.csv("ENCV_pred_test.csv")[[2]]) # Elastic net (cross validated for optimal lambda)
gb_test = as.vector(read.csv("GB_pred_test.csv")[[2]]) # Gradient boosting
ann_test = as.vector(read.csv("ANN_pred_test.csv")[[2]]) # Artificial neural network
xgb_test = as.vector(read.csv("XGB_pred_test.csv")[[2]]) # Extreme Gradient Boosting
credit_risk_test = read.csv("test_fs_ig.csv")$credit_risk # Actual target variable in test set


# Create dataframe with each base model's predicted probabilities of poor credit risk on test set
test_pred_df = data.frame(
  knn = knn_test,
  rf = rf_test,
  encv = encv_test,
  gb = gb_test,
  ann = ann_test,
  xgb = xgb_test,
  credit_risk = credit_risk_test
)

# Calculate predicted probabilities using meta model
test.pred = as.vector(predict(model, test_pred_df, type = "response")) #predicted probabilities

#Calculate AUC
library(pROC)
model.roc = roc(response = test_pred_df$credit_risk, 
                predictor = test.pred)
auc(model.roc) 

# Get optimal cutoff for events using ROC
opt_cutoff = coords(model.roc, "best", ret = "threshold")

library(caret)
# Classify events based on cutoff
test.class = ifelse(test.pred > opt_cutoff[[1]], 1, 0)
# Confusion matrix on test data
cm=confusionMatrix(table(test.class,test_pred_df$credit_risk), positive="1") 
# Calculate F1 score
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1