#Aim: reduce multicollinearity using filter-based feature selection 
#(computationally less expensive than other techniques (wrapper, hybrid))
library(mlr3)
library(mlr3filters)
library(FSelectorRcpp)
library(dplyr)

#Apply feature selection to JUST training set to prevent data leakage
train_full = read.csv("credit_train.csv")

cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration", "installment_rate", 
             "personal_status_sex", "other_debtors", "present_residence", 
             "property", "other_installment_plans","housing",
             "number_credits","job","people_liable","telephone",
             "foreign_worker","credit_risk")
train_full = train_full %>% mutate (across(all_of(cat_vars), as.factor))

#Turn this classification problem into a task object
task = TaskClassif$new(
  id="pred_risk", 
  backend = train_full, 
  target = "credit_risk")

#Use information gain to filter the data
filter = flt("information_gain")
feature_table = as.data.table(filter$calculate(task))

feature_score_df = data.frame(
  feature = feature_table$feature,
  score = feature_table$score
)
#sort in descending order and choose the top few features
feature_score_df = feature_score_df %>% arrange(desc(score))
topfeatures = head(feature_score_df, n = 16)$feature
#Create new training dataset - training, feature selection, information gain
train_fs_ig = train_full %>% select(any_of(topfeatures), credit_risk)

#Now, let's see if multicollinearity is reduced from before
full_model = glm(credit_risk ~., data = train_fs_ig, 
                 family = binomial(link="logit"))
library(car)
vifs = vif(full_model)
vifs
#Much better, now they're all less than 4 :) 

#Also select the same features from validation/testing sets
#read in CSV's
valid_full = read.csv("credit_valid.csv")
test_full = read.csv("credit_test.csv")
#Select features from each dataset
valid_fs_ig = valid_full %>% select(any_of(topfeatures), credit_risk)
test_fs_ig = test_full %>% select(any_of(topfeatures), credit_risk)
#Write train_fs_ig, valid_fs_ig, test_fs_ig, to new CSV files to be read in for creating classifier models
write.csv(train_fs_ig, "train_fs_ig.csv", row.names=FALSE)
write.csv(valid_fs_ig, "valid_fs_ig.csv", row.names=FALSE)
write.csv(test_fs_ig, "test_fs_ig.csv", row.names=FALSE)

