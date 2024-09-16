#Group26
#Yash Srivastava   PGID - 12010060
#Charika Bhatia    PGID - 12010036
#Jaideep Saraswat   PGID - 12010051

#Q1, Q2


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

#Reading the dataset
df <- read.csv("C:/Users/ysrivastava/SA4/SA4 Group Datasets/file26.csv")

#Checking for Null Values
sapply(df, function(x) sum(is.na(x)))
df <- drop_na(df)
df <- df[,c(-1,-2)] # Dropping "X" and "Customer_ID" for irrelevance.


#EDA
#Distribution of Churn
table(df$Churn)
#0    1 
#3662  1239
ggplot(df) + geom_bar(aes(x = Churn),width = 0.2)


#Distribution of Gender vs Churn
table(df$gender)
#Female   Male 
#2486   2505
#Almost same number of male and females.
table(df$Churn,df$Contract)
#      Female  Male
#No    1806    1856
#Yes    680    649
ggplot(df, aes(gender, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")
#Churning males are similar to churning females and vice versa
#Gender might not have a strong impact on Churning.


#Distribution of SeniorCitizen vs Churn
table(df$SeniorCitizen)
#0    1 
#4169  822
#Very less amount of senior citizen
table(df$Churn,df$SeniorCitizen)
#     0    1
#No  3179  483
#Yes  990  339
#If Senior citizen, then no. of churner-non-churners is close to each other, else there is significant difference
# between Churn-Not Churn
ggplot(df, aes(SeniorCitizen, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")



#Distribution of Partner vs Churn
table(df$Partner)
#No  Yes 
#2588 2403
table(df$Churn,df$Partner)
#     No  Yes
#No  1740 1922
#Yes  848  481
ggplot(df, aes(Partner, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of Dependents vs Churn
table(df$Dependents)
#No  Yes 
#3497 1494
table(df$Churn,df$Dependents)
#     No  Yes
#No  2402 1260
#Yes  1095  234
ggplot(df, aes(Dependents, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of tenure vs Churn
ggplot(df, aes(tenure, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")

#Distribution of PhoneService vs Churn
table(df$PhoneService)
#No  Yes 
#498 4493
table(df$Churn,df$PhoneService)
#     No  Yes
#No  372 3290
#Yes  126 1203
ggplot(df, aes(PhoneService, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of MultipleLines vs Churn
table(df$MultipleLines)
#No                No phone service              Yes 
#2384              498                           2109
table(df$Churn,df$MultipleLines)
#    No     No phone service   Yes
#No  1785   372                1505
#Yes  599   126                604
ggplot(df, aes(MultipleLines, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of InternetService vs Churn
table(df$InternetService)
#DSL         Fiber optic          No 
#1723        2193               1075                       
table(df$Churn,df$InternetService)
#    DSL         Fiber optic   No
#No  1391        1278         993
#Yes  332         915          82
ggplot(df, aes(InternetService, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of OnlineSecurity vs Churn
table(df$OnlineSecurity)
#No        No internet service           Yes 
#2502                1075                1414                       
table(df$Churn,df$OnlineSecurity)
#    No       No internet service  Yes
#No  1462          993             1207
#Yes 1040          82              207
ggplot(df, aes(OnlineSecurity, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of OnlineBackup vs Churn
table(df$OnlineBackup)
#No        No internet service           Yes 
#2185                1075                1731                       
table(df$Churn,df$OnlineBackup)
#    No     No internet service  Yes
#No  1316        993             1353
#Yes  869        82              378
ggplot(df, aes(OnlineBackup, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of DeviceProtection vs Churn
table(df$DeviceProtection)
#No       No internet service            Yes 
#2198                1075                1718                        
table(df$Churn,df$DeviceProtection)
#     No      No internet service  Yes
#No  1339          993             1330
#Yes  859          82              388
ggplot(df, aes(DeviceProtection, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of TechSupport vs Churn
table(df$TechSupport)
#No       No internet service            Yes 
#2454                1075                1462                        
table(df$Churn,df$TechSupport)
#     No      No internet service  Yes
#No  1429          993             1240
#Yes  1025          82              222
ggplot(df, aes(TechSupport, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of StreamingTV vs Churn
table(df$StreamingTV)
#No       No internet service            Yes 
#1986                1075                1930                        
table(df$Churn,df$StreamingTV)
#     No      No internet service  Yes
#No  1326         993             1342
#Yes  660          82              587
ggplot(df, aes(StreamingTV, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")



#Distribution of StreamingMovies vs Churn
table(df$StreamingMovies)
#No       No internet service            Yes 
#1983                1075                1933                      
table(df$Churn,df$StreamingMovies)
#     No      No internet service  Yes
#No  1308         993             1361
#Yes  675          82              572
ggplot(df, aes(StreamingMovies, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of Contract vs Churn
table(df$Contract)
#Month-to-month       One year       Two year 
#2772                 1031           1188                       
table(df$Churn,df$Contract)
#       Month-to-month   One year   Two year
#No            1596      913        1153
#Yes           1176      118        35
ggplot(df, aes(Contract, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")



#Distribution of PaperlessBilling vs Churn
table(df$PaperlessBilling)
#No  Yes 
#2036 2955                      
table(df$Churn,df$PaperlessBilling)
#     No  Yes
#No  1695 1967
#Yes  341  988
ggplot(df, aes(PaperlessBilling, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Distribution of PaymentMethod vs Churn
table(df$PaymentMethod)
#Bank transfer (automatic)   Credit card (automatic)          Electronic check              Mailed check 
#1102                           1077                          1677                          1135                      
table(df$Churn,df$PaymentMethod)
#           Bank transfer (automatic)    Credit card (automatic)    Electronic check     Mailed check
#No                        912                     914                 928                 908
#Yes                       190                     163                 749                 227
ggplot(df, aes(PaymentMethod, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")



#Distribution of MonthlyCharges vs Churn
ggplot(df, aes(MonthlyCharges, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")

#Distribution of TotalCharges vs Churn
ggplot(df, aes(TotalCharges, ..count..)) + geom_bar(aes(fill = Churn), position = "dodge")


#Splitting the dataset
set.seed(1235)
sample <- sample(c(TRUE, FALSE), nrow(df), replace = T, prob = c(0.6,0.4))
train_df <- df[sample, ]
test_df <- df[!sample, ]

#Data preparation
train_df$Partner <- factor(train_df$Partner, levels = c("No","Yes"), labels = c(0,1))
train_df$Dependents <- factor(train_df$Dependents, levels = c("No","Yes"), labels = c(0,1))
train_df$PhoneService <- factor(train_df$PhoneService, levels = c("Yes","No"), labels = c(1,0))
train_df$PaperlessBilling <- factor(train_df$PaperlessBilling, levels = c("No","Yes"), labels = c(0,1))
train_df <- dummy_cols(train_df)
write.csv(train_df, "x.csv") #Manipulating the dataset on excel
train_df <- read.csv("x.csv")
train_df$MonthlyCharges <- scale(train_df$MonthlyCharges)
train_df$TotalCharges <- scale(train_df$TotalCharges)
train_df$tenure <- scale(train_df$tenure)
train_df$Churn <- factor(train_df$Churn, levels = c("No","Yes"), labels = c(0,1))


#LDA
#Iteration1
model1.lda <- lda(Churn ~ .-Churn, data = train_df)
model1.lda


#Dropping correlated variables
train_df = subset(train_df, select = -c(MultipleLines_Yes, OnlineSecurity_Yes, OnlineBackup_Yes, DeviceProtection_Yes,  TechSupport_Yes , StreamingTV_Yes, StreamingMovies_Yes) )
train_df <- train_df[,c(-18)]

#Iteration2
model2.lda <- lda(Churn ~ .-Churn, data = train_df)
model2.lda

#Feature Selection using Greedy.Wilks
x = train_df$Churn
formulaAll=x~SeniorCitizen+Partner+Dependents+tenure+PhoneService+PaperlessBilling+MonthlyCharges+TotalCharges+gender_Male+InternetService_DSL+InternetService_Fiber_optic+Contract_Month.to.month+Contract_Two_year+PaymentMethod_Bank_transfer+PaymentMethod_Credit_card+PaymentMethod_Electronic_check
greedy.wilks(formulaAll,data=train_df, niveau = 0.15)


#Iteration3
model3.lda <- lda(Churn ~ Contract_Month.to.month+InternetService_Fiber_optic+tenure+PaymentMethod_Electronic_check+InternetService_DSL+TotalCharges+PaperlessBilling+SeniorCitizen+MonthlyCharges+PhoneService+Dependents, data = train_df)
model3.lda


#MOdel Prediction
#Model1
model1.pred <- predict(model1.lda, data = train_df)
lda_cm <- table(train_df$Churn, model1.pred$class)
lda_cm #Confusion Matrix
list(cm_model = lda_cm %>% prop.table() %>% round(3))
#Acuracy Rate
mean(model1.pred$class == train_df$Churn)
#error rate:
mean(model1.pred$class != train_df$Churn)


par(mfrow=c(1, 2))
#ROC Curve
prediction(model1.pred$posterior[,2], train_df$Churn) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()

#AUC
prediction(model1.pred$posterior[,2], train_df$Churn) %>%
  performance(measure = "auc") %>% .@y.values 

#Plotting discriminants
plot(model1.pred$x,xlab="first linear discriminant",col=factor(model1.pred$class))
legend("topleft", legend=levels(factor(model3.pred$class)), 
       text.col=seq_along(levels(factor(model3.pred$class))))


#Model2
model2.pred <- predict(model2.lda, data = train_df)
lda_cm <- table(train_df$Churn, model2.pred$class)
lda_cm #Confusion Matrix
list(cm_model = lda_cm %>% prop.table() %>% round(3))
#Acuracy Rate
mean(model2.pred$class == train_df$Churn)
#error rate:
mean(model2.pred$class != train_df$Churn)


par(mfrow=c(1, 2))
#ROC Curve
prediction(model1.pred$posterior[,2], train_df$Churn) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()

#AUC
prediction(model1.pred$posterior[,2], train_df$Churn) %>%
  performance(measure = "auc") %>% .@y.values 

#Plotting discriminants
plot(model1.pred$x,xlab="first linear discriminant",col=factor(model1.pred$class))
legend("topleft", legend=levels(factor(model3.pred$class)), 
       text.col=seq_along(levels(factor(model3.pred$class))))


#MOdel Diagnostics

#Hotelling's T square
#Model2
fit = hotelling.test(SeniorCitizen+Partner+Dependents+tenure+PhoneService+PaperlessBilling+MonthlyCharges+TotalCharges+gender_Male+InternetService_DSL+InternetService_Fiber_optic+Contract_Month.to.month+Contract_Two_year+PaymentMethod_Bank_transfer+PaymentMethod_Credit_card+PaymentMethod_Electronic_check~Churn, train_df)
fit

#Model3
fit = hotelling.test(Contract_Month.to.month+InternetService_Fiber_optic+tenure+PaymentMethod_Electronic_check+InternetService_DSL+TotalCharges+PaperlessBilling+SeniorCitizen+MonthlyCharges+PhoneService+Dependents~Churn, train_df,)
fit

#WilksLambda
#Model2
dat <- train_df[,c("SeniorCitizen","Partner","Dependents","tenure","PhoneService","PaperlessBilling","MonthlyCharges","TotalCharges","gender_Male","InternetService_DSL","InternetService_Fiber_optic","Contract_Month.to.month","Contract_Two_year","PaymentMethod_Bank_transfer","PaymentMethod_Credit_card","PaymentMethod_Electronic_check")]
dat <- as.matrix(dat, rownames.force = NA)
man <- manova(as.matrix(dat) ~ train_df$Churn)
summary(man, test="Wilks")

#Model3
dat <- train_df[,c("Contract_Month.to.month","InternetService_Fiber_optic","tenure","PaymentMethod_Electronic_check","InternetService_DSL","TotalCharges","PaperlessBilling","SeniorCitizen","MonthlyCharges","PhoneService","Dependents")]
dat <- as.matrix(dat, rownames.force = NA)
man <- manova(as.matrix(dat) ~ train_df$Churn)
summary(man, test="Wilks")


#Validating Model 3
model3.pred <- predict(model3.lda, newdata = test_df)
lda_cm <- table(test_df$Churn, model3.pred$class)
lda_cm #Confusion Matrix
list(cm_model = lda_cm %>% prop.table() %>% round(3))

#Acuracy Rate
mean(model3.pred$class == test_df$Churn)
#error rate
mean(model3.pred$class != test_df$Churn)


#Logistic Regression
#Iteration 1
model1.lr <- glm(Churn ~ .-Churn, family = "binomial", data = train_df)
summary(model1.lr)
vif(model1.lr)
write.csv(as.data.frame(summary(model1.lr)$coef), file="z.csv")


#Removing strongly correlated variables
train_df = subset(train_df, select = -c(MultipleLines_No, OnlineSecurity_No, OnlineBackup_No, DeviceProtection_No,  TechSupport_No , StreamingTV_No, StreamingMovies_No) )


#Model2
model2.lr <- glm(Churn ~ .-Churn, family = "binomial", data = train_df)%>%stepAIC(trace = FALSE)
summary(model2.lr)
vif(model2.lr)
probabilities <- model2.lr %>% predict(train_df, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, 1,0)
# Model accuracy
mean(predicted.classes==train_df$Churn)

model2.lr.pred <- predict(model2.lr, newdata = train_df, type = "response")
table(train_df$Churn, model2.lr.pred > 0.5)
list(
  model1 = table(train_df$Churn, model2.lr.pred > 0.5) %>% prop.table() %>% round(3)
)

#ROC Curve
par(mfrow=c(1, 2))
prediction(model2.lr.pred, train_df$Churn) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()

prediction(model2.lr.pred, train_df$Churn) %>%
  performance(measure = "auc") %>%
  .@y.values


#Model3
model3.lr <- glm(Churn ~ SeniorCitizen+tenure+PhoneService+PaperlessBilling+InternetService_Fiber_optic+InternetService_DSL+OnlineSecurity_Yes+TechSupport_Yes+StreamingTV_Yes+Contract_Month.to.month+Contract_Two_year+PaymentMethod_Electronic_check, family = "binomial", data = train_df)#%>%stepAIC(trace = FALSE)
summary(model3.lr)

vif(model3.lr)

#Confusion matrix
model3.lr.pred <- predict(model3.lr, newdata = train_df, type = "response")
table(train_df$Churn, model3.lr.pred > 0.5)
list(
  model1 = table(train_df$Churn, model3.lr.pred > 0.5) %>% prop.table() %>% round(3)
)


probabilities <- model3.lr %>% predict(train_df, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, 1,0)
# Model accuracy
mean(predicted.classes==train_df$Churn)

#ROC Curve
par(mfrow=c(1, 2))
prediction(model3.lr.pred, train_df$Churn) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()

prediction(model3.lr.pred, train_df$Churn) %>%
  performance(measure = "auc") %>%
  .@y.values

#Testing deviences
pchisq(2592.6, df = 2995, lower.tail = F)


#Model Diagnostics
#Model2
PseudoR2(model2.lr,which = "all") #Pseudo R squares
hoslem.test(model2.lr$y, fitted(model2.lr), g=10) #Hosmer-Lemeshow Goodness of Fit (GOF) Test
caret::varImp(model2.lr)

#Model3
PseudoR2(model3.lr,which = "all") #Pseudo R squares
hoslem.test(model3.lr$y, fitted(model3.lr), g=10) #Hosmer-Lemeshow Goodness of Fit (GOF) Test
caret::varImp(model3.lr)

#Chi square test find otimum model
anova(model2.lr,model3.lr, test = "Chisq")

#Model Validation
#Preping test dataset
test_df$Partner <- factor(test_df$Partner, levels = c("No","Yes"), labels = c(0,1))
test_df$Dependents <- factor(test_df$Dependents, levels = c("No","Yes"), labels = c(0,1))
test_df$PhoneService <- factor(test_df$PhoneService, levels = c("Yes","No"), labels = c(1,0))
test_df$PaperlessBilling <- factor(test_df$PaperlessBilling, levels = c("No","Yes"), labels = c(0,1))
test_df <- dummy_cols(test_df)
write.csv(test_df, "y.csv")
test_df <- read.csv("y.csv")
test_df$MonthlyCharges <- scale(test_df$MonthlyCharges)
test_df$TotalCharges <- scale(test_df$TotalCharges)
test_df$tenure <- scale(test_df$tenure)
test_df$Churn <- factor(test_df$Churn, levels = c("No","Yes"), labels = c(0,1))

#Validation
model3.lr.predT <- predict(model3.lr, newdata = test_df, type = "response")
table(test_df$Churn, model3.lr.predT > 0.5)
list(
  model1 = table(test_df$Churn, model3.lr.predT > 0.5) %>% prop.table() %>% round(3)
)

probabilities <- model3.lr %>% predict(test_df, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, 1,0)
# Model accuracy
mean(predicted.classes==test_df$Churn)





