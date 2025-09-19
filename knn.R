# Using K-nearest neighbors (KNN) to predict binary outcome: credit risk (good vs poor)

# Read in the data (cleaned + feature selection implemented)
train <- read.csv("train_fs_ig.csv")
valid <- read.csv("valid_fs_ig.csv")
test <- read.csv("test_fs_ig.csv")

# Change all categorical variables to factors
library(dplyr)
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker", "credit_risk")
train = train %>% mutate (across(all_of(cat_vars), as.factor))
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
test = test %>% mutate (across(all_of(cat_vars), as.factor))

# Standardize continuous variables since KNN is a distance-based algorithm
# Calculate mean and standard deviation from the training data
mean_dur = mean(train$duration)
sd_dur= sd(train$duration)
# Use training data statistics to standardize train, validation, and test sets
train$duration = (train$duration - mean_dur)/sd_dur
valid$duration = (valid$duration - mean_dur)/sd_dur
test$duration = (test$duration - mean_dur)/sd_dur

# Repeat the standardization process for the 'amount' variable
mean_amt = mean(train$amount)
sd_amt = sd(train$amount)
train$amount = (train$amount - mean_amt)/sd_amt
valid$amount = (valid$amount - mean_amt)/sd_amt
test$amount = (test$amount - mean_amt)/sd_amt

# Create dummy variables since KNN cannot handle factor variables directly
train.mat = model.matrix(credit_risk ~ ., data = train)[, -1]
valid.mat = model.matrix(credit_risk ~ ., data = valid)[, -1]
test.mat = model.matrix(credit_risk ~ ., data = test)[, -1]

# The knn() function requires dataframes without the target variable
train.dummies <- as.data.frame(train.mat)
valid.dummies <- as.data.frame(valid.mat)
test.dummies <- as.data.frame(test.mat)

# Loop through a range of K values to find the optimal number of neighbors
aucs = c()
neighbors = c()
library(class)
# Iterate through odd numbers for K to avoid ties
for (i in seq(3,499,4)){
  set.seed(2025)
  # Run KNN on validation data for the current K
  knn_valid = knn(train = train.dummies, 
                  test = valid.dummies, 
                  cl=train$credit_risk,
                  k = i,
                  prob = TRUE, 
                  use.all=FALSE)
  
  # Extract the proportion of votes for the winning class
  winner.probs = attr(knn_valid, "prob")
  # Convert to probabilities for the positive class ('1') for ROC analysis
  valid.pred = ifelse(knn_valid=="1", winner.probs, 1-winner.probs)
  
  # Calculate the AUC on the validation set and store the results
  library(pROC)
  knn_roc = roc(response = valid$credit_risk, predictor= valid.pred)
  auc = auc(knn_roc)
  aucs = c(aucs, auc)
  neighbors = c(neighbors, i)
}

# Create a data frame to view the AUC for each K value tested
knn_gridsearch = data.frame(
  AUC = aucs,
  K = neighbors
)

# Identify the K value that resulted in the highest AUC
best_k = knn_gridsearch$K[which.max(knn_gridsearch$AUC)]

# Rerun the model on the validation data using the optimal K
set.seed(2025)
knn_valid = knn(train = train.dummies, 
                test = valid.dummies, 
                cl=train$credit_risk,
                k = best_k,
                prob = TRUE, 
                use.all=FALSE)
# Get probabilities for the positive class
valid_winner.probs = attr(knn_valid, "prob")
valid.pred = ifelse(knn_valid=="1", valid_winner.probs, 1-valid_winner.probs)

# Check AUC on validation data with the optimal K
valid.roc = roc(response = valid$credit_risk, predictor= valid.pred)
auc(valid.roc)

# Evaluate the final model on the unseen test data using the optimal K
set.seed(2025)
knn_test = knn(train = train.dummies, 
               test = test.dummies, 
               cl=train$credit_risk,
               k = best_k,
               prob = TRUE, 
               use.all=FALSE)
# Get probabilities for the positive class
test_winner.probs = attr(knn_test, "prob")
test.pred = ifelse(knn_test=="1", test_winner.probs, 1-test_winner.probs) 

# Check AUC on test data
test.roc = roc(response = test$credit_risk, predictor= test.pred)
auc(test.roc)

# Identify optimal cutoff for classification based on the test ROC curve
library(caret)
opt_cutoff = coords(test.roc, "best", ret = "threshold")
test.class = ifelse(test.pred > opt_cutoff[[1]], 1, 0) 

# Confusion matrix on test data
cm=confusionMatrix(table(test.class,test$credit_risk), positive="1") 

# Calculating F1 score from the confusion matrix
prec = cm$byClass[['Precision']]
recall = cm$byClass[['Recall']]
f1 = 2*prec*recall/(prec+recall)
f1

# Write validation/test predictions to CSV files to later use as inputs for stacked classifier
write.csv(valid.pred, "KNN_pred_valid.csv")
write.csv(test.pred, "KNN_pred_test.csv")