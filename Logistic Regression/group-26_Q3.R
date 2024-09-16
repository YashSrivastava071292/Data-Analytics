#Group26
#Yash Srivastava   PGID - 12010060
#Charika Bhatia    PGID - 12010036
#Jaideep Saraswat   PGID - 12010051

#Q3


# Packages
library(tidyverse) #For data manipulation
library(dplyr) #For data manipulation
library(MASS)#Provides LDA and QDA functions
library(ggplot2) #Visualization
library(Boruta) # Feature selection
library(fastDummies) #Dummy variables
library(heplots) #Visualization
library(gridExtra) #Visualization
library(ROCR)# ROC curves
library("Hotelling")
library(klaR)
library(DescTools)
library(plyr)
library(modelr)
library(pscl)
library(broom)
library(ResourceSelection)
install.packages('OneR')
library(OneR)


#Creating category segment variable
df2 <- df
summary(df2$TotalCharges)
df2<- drop_na(df2)
df2['Customer_value_segment'] <- NA
df2<- df2 %>% mutate(Customer_value_segment = case_when(
  TotalCharges < 398.900 ~ "low",
  TotalCharges > 398.900 & TotalCharges < 3844.025 ~ "medium",
  TotalCharges > 3844.025 ~ "high"
))

#Preparing the dataset
df2$Partner <- factor(df2$Partner, levels = c("No","Yes"), labels = c(0,1))
df2$Dependents <- factor(df2$Dependents, levels = c("No","Yes"), labels = c(0,1))
df2$PhoneService <- factor(df2$PhoneService, levels = c("Yes","No"), labels = c(1,0))
df2$PaperlessBilling <- factor(df2$PaperlessBilling, levels = c("No","Yes"), labels = c(0,1))
df2$Churn <- factor(df2$Churn, levels = c("No","Yes"), labels = c(0,1))
df2 <- dummy_cols(df2)
write.csv(df2, "w.csv")
f2 <- read.csv("w.csv")
df2$MonthlyCharges <- scale(df2$MonthlyCharges)
df2$tenure <- scale(df2$tenure)

#Splitting dataset into test and train
set.seed(123)
sample <- sample(c(TRUE, FALSE), nrow(df2), replace = T, prob = c(0.6,0.4))
train_df2 <- df2[sample, ]
test_df2 <- df2[!sample, ]

#Model1
model_1 <- lda(Customer_value_segment ~ .-Customer_value_segment, data = train_df2)
model_1

model_1.pred <- predict(model_1, data = train_df2)
lda_cm <- table(train_df2$Customer_value_segment, model_1.pred$class)
lda_cm #Confusion Matrix
list(cm_model = lda_cm %>% prop.table() %>% round(3))
#Acuracy Rate
mean(model_1.pred$class == train_df2$Customer_value_segment)
#error rate
mean(model_1.pred$class != train_df2$Customer_value_segment)


#Removing strongly correlated variables
train_df3 = subset(train_df2, select = -c(MultipleLines_Yes,MultipleLines_No, OnlineSecurity_Yes,OnlineSecurity_No, OnlineBackup_Yes, OnlineBackup_No, DeviceProtection_Yes, DeviceProtection_No,  TechSupport_Yes, TechSupport_No , StreamingTV_Yes, StreamingTV_No, StreamingMovies_Yes, StreamingMovies_No) )

#Model2
model_2 <- lda(Customer_value_segment ~ .-Customer_value_segment, data = train_df3)
model_2


model_2.pred <- predict(model_2, data = train_df3)
lda_cm <- table(train_df3$Customer_value_segment, model_2.pred$class)
lda_cm #Confusion Matrix
list(cm_model = lda_cm %>% prop.table() %>% round(3))

#Acuracy Rate
mean(model_2.pred$class == train_df3$Customer_value_segment)
#error rate
mean(model_2.pred$class != train_df3$Customer_value_segment)

#Feature selection using greedy.wilks
x = train_df3$Customer_value_segment
formulaAll=x~SeniorCitizen+Partner+Dependents+tenure+PhoneService+PaperlessBilling+MonthlyCharges+TotalCharges+gender_Male+InternetService_DSL+InternetService_Fiber_optic+Contract_Month.to.month+Contract_Two_year+PaymentMethod_Bank_transfer+PaymentMethod_Credit_card+PaymentMethod_Electronic_check+Churn_1 
greedy.wilks(formulaAll,data=train_df3, niveau = 0.15)
a$results
write.csv(as.data.frame(a$results), file="x.csv")

#Model3
model_3 <- lda(Customer_value_segment ~ tenure+TotalCharges+MonthlyCharges+Contract_Two_year+Churn_1+InternetService_Fiber_optic+InternetService_DSL+PhoneService+PaymentMethod_Credit_card, data = train_df3)
model_3


model_3.pred <- predict(model_3, data = train_df3)
lda_cm <- table(train_df3$Customer_value_segment, model_3.pred$class)
lda_cm #Confusion Matrix
list(cm_model = lda_cm %>% prop.table() %>% round(3))


mean(model_3.pred$class == train_df3$Customer_value_segment)
#Acuracy Rate: 0.8780326
mean(model_2.pred$class != train_df3$Customer_value_segment)
#error rate: 0.1219674


#Diagnostic
#Hotelling's T square
#Model2
fit = hotelling.test(SeniorCitizen+Partner+Dependents+tenure+PhoneService+PaperlessBilling+MonthlyCharges+TotalCharges+gender_Male+InternetService_DSL+InternetService_Fiber_optic+Contract_Month.to.month+Contract_Two_year+PaymentMethod_Bank_transfer+PaymentMethod_Credit_card+PaymentMethod_Electronic_check+Churn_1~Customer_value_segment, train_df3,)
fit

#WilksLambda
#Model2
dat <- train_df3[,c("SeniorCitizen","Partner","Dependents","tenure","PhoneService","PaperlessBilling","MonthlyCharges","TotalCharges","gender_Male","InternetService_DSL","InternetService_Fiber_optic","Contract_Month.to.month","Contract_Two_year","PaymentMethod_Bank_transfer","PaymentMethod_Credit_card","PaymentMethod_Electronic_check","Churn_1")]
dat <- as.matrix(dat, rownames.force = NA)
man <- manova(as.matrix(dat) ~ train_df3$Customer_value_segment)
summary(man, test="Wilks")

#Model3
#Hotelling's T square
fit = hotelling.test(tenure+TotalCharges+MonthlyCharges+Contract_Two_year+Churn_1+InternetService_Fiber_optic+InternetService_DSL+PhoneService+PaymentMethod_Credit_card~Customer_value_segment, train_df3,)
fit

#WilksLambda
#Model3
dat <- train_df3[,c("tenure","TotalCharges","MonthlyCharges","Contract_Two_year","Churn_1","InternetService_Fiber_optic","InternetService_DSL","PhoneService","PaymentMethod_Credit_card")]
dat <- as.matrix(dat, rownames.force = NA)
man <- manova(as.matrix(dat) ~ train_df3$Customer_value_segment)
summary(man, test="Wilks")