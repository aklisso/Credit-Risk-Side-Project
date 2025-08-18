#data: https://www.kaggle.com/datasets/varunchawla30/german-credit-data
#Originally sourced from UCI, but was modified to be more easily readable

#Read in data
credit <- read.csv("german_credit_data.csv")

#Renaming columns to American names (they are in German) based on data dictionary on Kaggle
#Note: DM = Deutsche Mark (German currency)

library(dplyr)
credit = credit %>% rename (checkings = laufkont,
                            duration = laufzeit, #duration of loan in months
                            credit_history = moral,
                            purpose = verw,
                            amount = hoehe,
                            savings = sparkont,
                            employment_duration = beszeit,
                            installment_rate = rate,
                            personal_status_sex = famges, #I might want to separate these
                            other_debtors = buerge,
                            present_residence = wohnzeit,
                            property = verm,
                            age = alter,
                            other_installment_plans = weitkred,
                            housing = wohn,
                            number_credits = bishkred,
                            job = beruf,
                            people_liable = pers,
                            telephone = telef,
                            foreign_worker = gastarb,
                            credit_risk = kredit #target variable
)

#Changing categorical variables to factors
cat_vars = c("checkings", "credit_history", "purpose", "savings", 
             "employment_duration", "installment_rate", 
             "personal_status_sex", "other_debtors", "present_residence", 
             "property", "other_installment_plans","housing",
             "number_credits","job","people_liable","telephone",
             "foreign_worker","credit_risk")
credit = credit %>% mutate (across(all_of(cat_vars), as.factor))

#right now installment rate is ordered from highest to lowest, but I won't switch it because then I can't refer to the data
# dictionary on Kaggle

#univariate EDA on numerical variables
library(ggplot2)
ggplot(credit, aes(x=duration)) + geom_histogram()
#Duration is right skewed, most appear around 12-24 months

ggplot(credit, aes(x=amount)) + geom_histogram()
#amount is right skewed; mostly 200-ish DM

ggplot(credit, aes(x=age)) + geom_histogram()
#age is also right-skewed, with most being in late 20's

#Since the numerical variables are right-skewed, we will probably have to do nonparametric tests.
#Note: plotting the log of each of these variables makes them look less skewed. Might come in handy later.

#Determine frequency of missing values for each variable
sum(is.na(credit)) #no missing values :)


summary(credit)
#Checkings: the fewest people have value 3 (0-200 DM); it's pretty common to either have NO checkings or a year's salary/>2000 DM
#Credit history: the majority of people have no credits taken out, or all paid back duly, second common: all credits at THIS BANK paid back
#Purpose: Mostly 3 (furniture/equipment); the least is repairs
#Savings: vast majority is unknown/no savings account (weird?)
#Employment duration: mostly 1-4 years, also >=7 years
#Installment rate: majority 4: <20% of disposable income
#Personal status/sex: least commonly divorced men; MOSTLY married or widowed men
#Mostly no other debtors
#Other installment plans: least commonly at stores
#Housing: mostly renting
# Number of credits: mostly 1
#Job: vast majority skilled/employed
#People liable: Mostly 2 (0-2)
#Telephone roughly even-ish
#Foreign worker: mostly not foreign
#Credit risk: 70-30 split- not too bad! Might need to oversample.


#Split into training/validation/testing before examining bivariate relationships
# 60-30-10 split because we have a good amount of data, but not a ton
# My statistics professor has recommended 70-20-10 for large datasets, and 50-40-10 for smaller
# So this is a middle ground
library(splitTools)
set.seed(2025)
partition = c(train = 0.6, valid = 0.3, test = 0.1)
inds = partition(credit$credit_risk, p = partition)
inds
train = credit[inds$train,]
valid = credit[inds$valid,]
test = credit [inds$test, ]

#Running logistic regression ONLY to get GVIF
full_model = glm(credit_risk ~., data = train, 
                  family = binomial(link="logit"))
library(car)
vifs = vif(full_model)
#purpose, employment duration, property, housing are all > 4
#a value >5 indicates multicollinearity, so we will be wary of these.
#Regularized regression might be more appropriate than logistic for this data
lasso = glmnet(x = train_x, )
#use this code: https://bookdown.org/tpinto_home/Regularisation/lasso-regression.html
