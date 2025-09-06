#K nearest neighbors

#Read in CSV's created in cleaning_eda.R
train <- read.csv("train_fs_ig.csv")
valid <- read.csv("valid_fs_ig.csv")
test <- read.csv("test_fs_ig.csv")

#Factorize categorical variables
library(dplyr)
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration",  
             "personal_status_sex", "other_debtors",
             "property", "other_installment_plans","housing",
             "foreign_worker", "credit_risk")
train = train %>% mutate (across(all_of(cat_vars), as.factor))
valid = valid %>% mutate (across(all_of(cat_vars), as.factor))
test = test %>% mutate (across(all_of(cat_vars), as.factor))

#scale numerical variables
#standardize duration
mean_dur = mean(train$duration)
sd_dur= sd(train$duration)
#use values in training data to standardize validation/test data
train$duration = (train$duration - mean_dur)/sd_dur
valid$duration = (valid$duration - mean_dur)/sd_dur
test$duration = (test$duration - mean_dur)/sd_dur
#repeat for amt
mean_amt = mean(train$amount)
sd_amt = sd(train$amount)
train$amount = (train$amount - mean_amt)/sd_amt
valid$amount = (valid$amount - mean_amt)/sd_amt
test$amount = (test$amount - mean_amt)/sd_amt

#Turn categorical variables into dummies since KNN cannot handle factors
train.mat = model.matrix(credit_risk ~ ., data = train)[, -1]
valid.mat = model.matrix(credit_risk ~ ., data = valid)[, -1]
test.mat = model.matrix(credit_risk ~ ., data = test)[, -1]

#Dataframes for KNN should not contain the target variable
train.dummies <- as.data.frame(train.mat)
valid.dummies <- as.data.frame(valid.mat)
test.dummies <- as.data.frame(test.mat)

aucs = c()
neighbors = c()
library(class)
for (i in seq(3,499,4)){ #anything above 503 gives "error: too many ties"
set.seed(2025)
knn_valid = knn(train = train.dummies, 
                test = valid.dummies, 
                cl=train$credit_risk,
                k = i, #iterates through different values
                prob = TRUE, 
                use.all=FALSE)
winner_probs = attr(knn_valid, "prob") # pull out "probability" of being in the selected class (either 0 or 1)
valid_probs = ifelse(knn_valid=="1", winner_probs, 1-winner_probs) #if predicted class is 0, want 1-pr(0) to get pr(1)

library(pROC)
knn_roc = roc(response = valid$credit_risk, predictor= valid_probs)
auc = auc(knn_roc)
aucs = c(aucs, auc)
neighbors = c(neighbors, i)
}
knn_gridsearch = data.frame(
  AUC = aucs,
  K = neighbors
)
View(knn_gridsearch)

knn_gridsearch$K[which.max(knn_gridsearch$AUC)] #optimal K is 27

#rerun with optimal K
set.seed(2025)
knn_valid = knn(train = train.dummies, 
                test = valid.dummies, 
                cl=train$credit_risk,
                k = 27, #iterates through different values
                prob = TRUE, 
                use.all=FALSE)
winner_probs = attr(knn_valid, "prob") # pull out "probability" of being in the selected class (either 0 or 1)
valid_probs = ifelse(knn_valid=="1", winner_probs, 1-winner_probs) #if predicted class is 0, want 1-pr(0) to get pr(1)
#AUC
knn_roc = roc(response = valid$credit_risk, predictor= valid_probs)
auc(knn_roc)

#confusion matrix
library(caret)
opt_cutoff = coords(knn_roc, "best", ret = "threshold")
knn_valid_class = ifelse(valid_probs > opt_cutoff[[1]], 1, 0) 
confusionMatrix(table(knn_valid_class,valid$credit_risk), positive="1") 
