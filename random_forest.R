#Random forest on filtered data

#Read in the feature-selected data
train = read.csv("train_fs_ig.csv")

#Convert categorical variables to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration", "installment_rate", 
             "personal_status_sex", "other_debtors", "present_residence", 
             "property", "other_installment_plans","housing",
             "job","foreign_worker","credit_risk")
train = train %>% mutate (across(all_of(cat_vars), as.factor))

library(randomForest) #most well-known implementation of random forest in R

set.seed(2025)
model = randomForest(
  formula = credit_risk~.,
  data = train
)

tail(model$err.rate) #error looks weirdly high for nonevents

#AUC on validation data