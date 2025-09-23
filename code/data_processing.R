#Aim: implement filter-based feature selection to replicate journal article 
# (https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00882-0) 

library(mlr3)
library(mlr3filters)
library(FSelectorRcpp)
library(dplyr)

# We will determine which features to use based solely on the training set to prevent data leakage, 
#then apply to validation/testing
train_full = read.csv("credit_train.csv")
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration", "installment_rate", 
             "personal_status_sex", "other_debtors", "present_residence", 
             "property", "other_installment_plans","housing",
             "number_credits","job","people_liable","telephone",
             "foreign_worker","credit_risk")
train_full = train_full %>% mutate (across(all_of(cat_vars), as.factor))

# Perform another check to see if there's multicollinearity
full_model = glm(credit_risk ~., data = train_fs_ig, 
                 family = binomial(link="logit"))
library(car)
vifs = vif(full_model)
vifs

# No multicollinearity is present, but I will implement feature selection anyway because it's interesting and I want
# to gain experience with it

# Turn this classification problem into a task object
task = TaskClassif$new(
  id="pred_risk", 
  backend = train_full, 
  target = "credit_risk")

# Use this information gain to filter the data
filter = flt("information_gain")
feature_table = as.data.table(filter$calculate(task))

feature_score_df = data.frame(
  feature = feature_table$feature,
  score = feature_table$score
)
# Sort in descending order of information gain and choose the top few features
feature_score_df = feature_score_df %>% arrange(desc(score))
topfeatures = head(feature_score_df, n = 13)$feature

# Variables that were filtered out (excluding credit_risk):
setdiff(colnames(train_full), topfeatures)

# Create new training dataset with just the selected features
train_fs_ig = train_full %>% dplyr::select(any_of(topfeatures), credit_risk)

# Also select the same features from validation/testing sets:

# Read in CSV files
valid_full = read.csv("credit_valid.csv")
test_full = read.csv("credit_test.csv")

# Select features from each dataset
valid_fs_ig = valid_full %>% dplyr::select(any_of(topfeatures), credit_risk)
test_fs_ig = test_full %>% dplyr::select(any_of(topfeatures), credit_risk)

#Write train_fs_ig, valid_fs_ig, test_fs_ig, to new CSV files to be read in for creating classifier models

write.csv(train_fs_ig, "train_fs_ig.csv", row.names=FALSE)
write.csv(valid_fs_ig, "valid_fs_ig.csv", row.names=FALSE)
write.csv(test_fs_ig, "test_fs_ig.csv", row.names=FALSE)

