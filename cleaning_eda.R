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

table(credit$credit_risk)
#Credit Risk is 0=poor, 1=good. I would think intuitively that 1 is the event (poor), so let's switch them.
credit$credit_risk= if_else(credit$credit_risk==0, 1, 0)


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


#Collapse factors in "purpose" since 4/5, 8/10 have very few
table(credit$purpose) #looks like nobody even has education as a value
#First, Recode the factor, otherwise the numbers will get all messed up
library(forcats)
credit$purpose = fct_recode(credit$purpose,
                            "other" = "0",
                            "car_new" = "1",
                            "car_used" = "2",
                            "furniture_equipment" = "3",
                            "radio_television" = "4",
                            "domestic_appliances" = "5",
                            "repairs" = "6",
                            "vacation" = "8",
                            "retraining" = "9",
                            "business" = "10"
)
table(credit$purpose) #check it's right
credit$purpose = fct_collapse(credit$purpose,
                              "electronics_appliances" = c("radio_television","domestic_appliances"),
                              "travel" = c("vacation","retraining"))
table(credit$purpose) #check it's right

table(credit$number_credits) #could collapse 3, 4, since 4 has so few
credit$number_credits = fct_recode(credit$number_credits,
                                   "1" = "1",
                                   "2_to_3" = "2",
                                   "4_to_5" = "3",
                                   "6_or_more" = "4")
credit$number_credits = fct_collapse(credit$number_credits,
                                     "4_or_more" = c("4_to_5", "6_or_more"))
table(credit$number_credits) #verifying it was done correctly


#Split into training/validation/testing before examining bivariate relationships
# 60-30-10 split because we have a good amount of data, but not a ton
# My statistics professor has recommended 70-20-10 for large datasets, and 50-40-10 for smaller
# So this is a middle ground
library(splitTools)
set.seed(2025)
split = c(train = 0.6, valid = 0.3, test = 0.1)
#Stratified split based on outcome
inds = splitTools::partition(credit$credit_risk, split) #must specificy splitTools due to conflict w/ mlr3 package used later
inds
train = credit[inds$train,]
valid = credit[inds$valid,]
test = credit [inds$test, ]

#Running logistic regression ONLY to get GVIF
full_model = glm(credit_risk ~., data = train, 
                  family = binomial(link="logit"))
library(car)
vifs = vif(full_model)
#employment duration, property, housing are all > 4. But collapsing categories isn't necessary.
table(credit$employment_duration)
table(credit$property)
table(credit$housing)

#According to online sources a value >4 may indicate multicollinearity, so we will address this later.

#Write cleaned data to CSV
write.csv(credit, "credit_cleaned.csv", row.names=FALSE)
write.csv(train, "credit_train.csv", row.names = FALSE)
write.csv(valid, "credit_valid.csv", row.names = FALSE)
write.csv(test, "credit_test.csv", row.names = FALSE)


#***THERE ARE SPACES IN THE VARIABLE VALUES FOR PURPOSE - MUST FIX THIS
#***For PURPOSE get rid of all funky characters like parentheses, spaces, slash, etc.





