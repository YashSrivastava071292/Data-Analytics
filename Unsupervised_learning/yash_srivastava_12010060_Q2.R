#Unsupervised Learning - Individual Assignment 1
#Name: Yash Srivastava
#Email: yash_srivastava_ampba2021s@isb.edu
#PGID: 12010060


library(fastDummies)
library(corrplot)
library(ggplot2)

#Reading the dataset
df_cars <- read.csv("ToyotaCorolla.csv")
str(df_cars)
write.csv(names(df_cars), file="names.csv")
df_cars$Cylinders
#A)
#Categorical vars: 
#"Model"             "Fuel_Type"         "Met_Color"         "Color"             "Automatic"        
#"Doors"             "Cylinders"         "Gears"             "Mfr_Guarantee"     "Dealer_Guarantee"  
#"Guarantee_Period"  "ABS"               "Airbag_1"          "Airbag_2"         "Aircond"           
#"Automatic_aircond" "Boardcomputer"     "CD_Player"         "Central_Lock"      "Powered_Windows"  
#"Power_Steering"    "Radio"             "Mistlamps"         "Sport_Model"       "Backseat_Divider"  "Metallic_Rim"

#B)
#Dummy variables, also know as indicator variables represent 2 more categories of an attribute. Let us say a variable has
#N categories, then we can create either N or N-1 dummy variables. Each dummy variable will tell weather category is present or not.

#Eg: the variable Gears has 2 categories 5 and 6. So its dummy variable will be Gears_5, Gears_6 and both will contain oly 0 and 1.
#1 representing either of the category present.

#c) For the categorical variables with N categories we can create N or N-1 dummy variables.
# For variables with only 2 categories we can create N-1 dummy variable which will contain the information of the variable.
#Some some cases creating N variables will result in redundant information for the Nth variable.
#Eg: for this variable "Aircond", the car either has it or it doesnt so this variable can be expressed as 0 and 1.

#d)
#We are using the package fastdummies to create dummy variables.

new_dummy <- dummy_cols(df_cars, select_columns = c("Mfg_Year","Fuel_Type","Color","Doors","Gears","Guarantee_Period"))
#Remaining variablea are already with values 0 and 1

new_dummy_df <- new_dummy[,-c(6,8,11,14,16,21)]
names(new_dummy_df)
write.csv(head(new_dummy_df), file="dumy.csv")
#e)
#correlation matrix for continous variables
#dataset: 
df_contvar <- df_cars[,c("Price","Age..month.","KM","HP","CC","Quarterly_Tax","Weight")]
write.csv(cor(df_contvar), file = "corr.csv")
cor(df_contvar)
M<-cor(df_contvar)

# Price has a very strong negative correlation to Age variable which is fairly obvious since the we are predicting
#price of used cars. So if the car is used for a less amount of time then its resale value will be high.

#Plotting for correlation matrix
corrplot(M, method = "number")

#Scatterplot between price and age
ggplot(df_contvar, aes(x=Age..month., y=Price)) + geom_point() +  
  labs(subtitle="Price vs Age..month.", y="Price", x="Age..month.", title="Scatterplot")
cor(df_contvar$Price, df_contvar$Age..month.)#-0.8765905

#Scatterplot between price and Kilometers driven
ggplot(df_contvar, aes(x=KM, y=Price)) + geom_point() +  
  labs(subtitle="Price vs KM", y="Price", x="KM", title="Scatterplot")
cor(df_contvar$Price, df_contvar$KM)#-0.5699602   Not a very strong correlation.


#Scatterplot between price and Weight
ggplot(df_contvar, aes(x=Weight, y=Price)) + geom_point() +  
  labs(subtitle="Price vs Weight", y="Price", x="Weight", title="Scatterplot")
cor(df_contvar$Price, df_contvar$Weight)#0.5811976   Not a very strong correlation.


#Scatterplot between Age and Kilometer
ggplot(df_contvar, aes(x=KM, y=Age..month.)) + geom_point() +  
  labs(subtitle="Age..month. vs KM", y="Age..month.", x="KM", title="Scatterplot")
cor(df_contvar$Age..month., df_contvar$KM)#0.5056722


#Correlation matrix for all variables
names(df_cars)
C<-cor(df_cars[,-c(1,2,5,6,8,11,14,15,16,21)])
corrplot(C, method = "number" )
#write.csv(cor(df_cars[,-c(1,2,5,6,8,11,14,15,16,21)]), file = "cor.csv")
