library("imputeTS")
library("forecast")
library("fpp2")
library("wavelets")
library("zoo")
library("ggpubr")
library("TSA")
library("xts")
library("timeSeries")
library("vcdExtra")
library("dygraphs")
library("graphics")
library("tseries")
library ("urca")
library(readxl)
library(xlsx)

my_data <- read_excel("SouvenirSales.xlsx")

#Q1 a)------------------------------------------------------------------------------------------
my_data <- ts(my_data$Sales, start = c(1995,1), frequency = 12)
autoplot(my_data) + xlab("Year") +
  ylab("Thousands") +  ggtitle("Monthly Sales of Souvenir at a shop in New York") + theme(plot.title = element_text(hjust = 0.5))

#---------------------------------------------------------------------------------------------

#Splitting the dataset into Train and Test
train <- window(my_data,end=c(2000,12), frequency=12)
test <- window(my_data,start=c(2001,1), frequency=12)

autoplot(train)
autoplot(test)

#Q1 b,d,e)----------------------------------------------------------------------------------------------------
#Fit a linear trend model with additive seasonality (Model A)
fit <- tbats(train) ## check if data has seasonality
seasonal <- !is.null(fit$seasonal)
seasonal

train_lt <- tslm(train ~ trend + season)  ## Building Linear Trend
summary(train_lt)
write.csv(as.data.frame(summary(train_lt)$coef), file="x.csv")

train_lt_pred <- forecast(train_lt, h=length(train ), level = 0) #Forecasting on Train
summary(train_lt_pred)

train_lt_pred_Test <- forecast(train_lt, h=length(test ), level = 0) #Forecasting on Validation
summary(train_lt_pred_Test)


#Fit a Exponential trend model with multiplicative seasonality (Model B)
train_et <- tslm(train ~ trend + season, lambda = 0)  ## Building Exponential Trend
summary(train_et)
write.csv(as.data.frame(summary(train_et)$coef), file="x.csv")

train_et_pred <- forecast(train_et, h=length(train ), level = 0) #Forecasting on Train
summary(train_et_pred)

train_et_pred_Test <- forecast(train_et, h=length(test ), level = 0) #Forecasting on Validation
summary(train_et_pred_Test)


accuracy(train_lt_pred$mean,test)
accuracy(train_et_pred$mean,test)


#Q1 c)--------------------------------------------------------------------------------------


plot(train , xlab ="Time", ylab= "Sales", bty="l" )#Actual vs fitted
lines(train_lt$fitted,lwd=2, col="blue")

plot(train_lt_pred, xlab ="Time", ylab= "Sales", flty=2, bty="l")#Predicted Training
lines(train_et_pred$mean,lwd=2, col="red")

plot(train_lt_pred_Test, xlab ="Time", ylab= "Sales", flty=2, bty="l")#Predicted Validation set
plot(train_lt_pred_Test$residuals, main= "Residual Plot", ylab= "Residuals", col="blue")#Residual plot


plot(train , xlab ="Time", ylab= "Sales", bty="l" )#Actual vs fitted
lines(train_et$fitted,lwd=2, col="blue")

plot(train_et_pred, xlab ="Time", ylab= "Sales", flty=2, bty="l", ylim=c(0,80000)) #Predicted Training
plot(train_et_pred_Test, xlab ="Time", ylab= "Sales", flty=2, bty="l", ylim=c(0,80000))#Predicted Validation set
plot(train_et_pred_Test$residuals, main= "Residual Plot", ylab= "Residuals", col="red")#Residual plot


#Q1 f)---------------------------------------------------------------------------------------
#Forecasting for Jan 2002
Exp_forcast <- tslm(my_data ~ trend+season, lambda = 0)
forecast(Exp_forcast, h=1, level = 0)


#Q1 g)---------------------------------------------------------------
#ACF and PACF Plots
plot(acf(train_et$residuals, lag.max = 20),
     main = "Autocorrelation",
     xlab = "Lag",
     ylab = "ACF")

plot(pacf(train_et$residuals, lag.max = 20),
     main = "Autocorrelation",
     xlab = "Lag",
     ylab = "PACF")

#Q1 h)---------------------------------------------------------------
#Fit AR Model
error1 =train_et$residuals
train_arima <- Arima(error1, order = c(2,0,0))
summary(train_arima )
automodel <- auto.arima(train)
automodel


#Q1 i)---------------------------------------------------------------
#Forecasting for Jan 2002 using Regression and AR Model
error2 = Exp_forcast$residuals
train_arima2 <- Arima(error2, order = c(2,0,0))
summary(train_arima2)

forecast.auto <- forecast(train_arima2,h=1)#0.06501293
#Ans = 13484 + 0.06501293
