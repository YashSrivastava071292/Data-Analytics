#Group26
#Yash Srivastava   PGID - 12010060
#Charika Bhatia    PGID - 12010036
#Jaideep Saraswat   PGID - 12010051

#Q4, Q5, Q6


#packages
library(survival)
library(survminer)

#Reading the dataset
df <- read.csv("C:/Users/ysrivastava/SA4/SA4 Group Datasets/file26.csv")

#Checking for Null values
sapply(df, function(x) sum(is.na(x)))
df <- drop_na(df)
df <- df[,c(-1,-2)] # Dropping "X" and "Customer_ID" for irrelevance.
df$Churn<- ifelse(df$Churn=="Yes", 1,0)
attach(df)

#Survival curve
surCurve1 <- survfit(Surv(tenure, Churn)~1, data=df)
ggsurvplot(surCurve1)
summary(surCurve1, times = seq(0,80,12))

#Survival curve for gender
surCurve2 <- survfit(Surv(tenure, Churn)~gender, data=df)
ggsurvplot(surCurve2)

#Survival curve for PhoneService
surCurve3 <- survfit(Surv(tenure, Churn)~PhoneService, data=df)
ggsurvplot(surCurve3)

#Survival curve for SeniorCitizen
surCurve4 <- survfit(Surv(tenure, Churn)~SeniorCitizen, data=df)
ggsurvplot(surCurve4)

#Survival curve for Partner
surCurve5 <- survfit(Surv(tenure, Churn)~Partner, data=df)
ggsurvplot(surCurve5)

#Survival curve for Dependents
surCurve6 <- survfit(Surv(tenure, Churn)~Dependents, data=df)
ggsurvplot(surCurve6)

#Survival curve for Contract
surCurve7 <- survfit(Surv(tenure, Churn)~Contract, data=df)
ggsurvplot(surCurve7)

#Survival curve for PaperlessBilling
surCurve7 <- survfit(Surv(tenure, Churn)~PaperlessBilling, data=df)
ggsurvplot(surCurve7)

#Survival curve for InternetService
surCurve8 <- survfit(Surv(tenure, Churn)~InternetService, data=df)
ggsurvplot(surCurve8)

#Survival curve for StreamingTV
surCurve9 <- survfit(Surv(tenure, Churn)~StreamingTV, data=df)
ggsurvplot(surCurve9)

#Survival curve for StreamingMovies
surCurve10 <- survfit(Surv(tenure, Churn)~StreamingMovies, data=df)
ggsurvplot(surCurve10)

#Survival curve for TechSupport
surCurve11 <- survfit(Surv(tenure, Churn)~TechSupport, data=df)
ggsurvplot(surCurve11)

#Survival curve for DeviceProtection
surCurve12 <- survfit(Surv(tenure, Churn)~DeviceProtection, data=df)
ggsurvplot(surCurve12)

#Survival curve for OnlineBackup
surCurve13 <- survfit(Surv(tenure, Churn)~OnlineBackup, data=df)
ggsurvplot(surCurve13)

#Survival curve for OnlineSecurity
surCurve14 <- survfit(Surv(tenure, Churn)~OnlineSecurity, data=df)
ggsurvplot(surCurve14)

#Survival curve for PaymentMethod
surCurve15 <- survfit(Surv(tenure, Churn)~PaymentMethod, data=df)
ggsurvplot(surCurve15)

#Survival curve for PaperlessBilling
surCurve16 <- survfit(Surv(tenure, Churn)~PaperlessBilling, data=df)
ggsurvplot(surCurve16)

#Cox Hazard Proportionality Model
#Model1
cox.mod <- coxph(Surv(tenure,Churn)~gender+PhoneService+SeniorCitizen+Partner+Dependents+MultipleLines+Contract+PaperlessBilling+InternetService+StreamingTV+StreamingMovies+TechSupport+DeviceProtection+OnlineBackup+OnlineSecurity+PaymentMethod+MonthlyCharges+TotalCharges)
summary(cox.mod)

#Removing correlated variables
df = subset(df, select = -c(MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,  TechSupport, StreamingMovies) )

#Model2
cox.mod2 <- coxph(Surv(tenure,Churn)~gender+PhoneService+SeniorCitizen+Partner+Dependents+Contract+PaperlessBilling+InternetService+StreamingTV+PaymentMethod+MonthlyCharges+TotalCharges)
summary(cox.mod2)

#Keeping variables which had significant differences among groups on the basis of survival curves
#Model3
cox.mod3 <- coxph(Surv(tenure,Churn)~+Partner+Contract+PaperlessBilling+InternetService+PaymentMethod+MonthlyCharges+TotalCharges, data = df)
summary(cox.mod3)

#Testing for Optimum model using Liklihood ratio test
anova(cox.mod2, cox.mod3, test="LRT")

#Plotting Survival Curve for Optimum Cox Hazard model:
ggsurvplot(survfit(cox.mod3, data=df), color = "#2E9FDF",
           ggtheme = theme_minimal())