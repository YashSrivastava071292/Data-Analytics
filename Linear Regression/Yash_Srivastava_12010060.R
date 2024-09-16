#Libraries required

library(gplots)
library("corrplot")
library("ggcorrplot")
library(xtable)
library(dplyr)
library(ggplot2)
library(car)
library(leaps)
library(broom)
library(Boruta)
library(fastDummies)
library(psych)

#---------------------------------------------------------------------------------------------------------------

#import dataset
houseData <- read.csv("house.csv")
df <- data.frame(houseData)
summary(df)
head(df)
str(df)

#---------------------------------------------------------------------------------------------------------------

#Look for null values
summary(df)
head(df)
sapply(df,function(x) sum(is.na(x)))
#FALSE
#Our dataset does not have any null values
names(df)
options(scipen=999)
#---------------------------------------------------------------------------------------------------------------

#Continuous Variables:  sqft_lot15, sqft_living15, sqft_basement, sqft_above, sqft_lot, sqft_living
#Discrete variables: grade, condition, view, waterfront, floors, bathrooms, bedrooms

#---------------------------------------------------------------------------------------------------------------

# EDA on the dataset to find
# 1. Outlier detection
# 2. Relationship between Price and independent variables
# 3. Correlation matrix
# 4. Checking for zeros concentration

#---------------------------------------------------------------------------------------------------------------

# Flag house renovation
df$is_renovated <- ifelse(df$yr_renovated == 0,0,1)
df$is_renovated
# Calculating house age
df$house_age <- ifelse(df$yr_renovated == 0, 2020 - df$yr_built, 2020 - df$yr_renovated)
df$house_age
cor(df$house_age,df$price)
#-0.071    So house age not that strongly correlated with Price
Agewise_price <- df %>% select(house_age,price) %>% group_by(house_age) %>% summarise(mean_price = mean(price))
ggplot(Agewise_price, aes(x=house_age, y=mean_price)) + geom_bar(stat="identity", width=.7, fill="blue")

# Type of House: Multistory / Ground only
df$type_livingsqft <- ifelse(df$sqft_above < df$sqft_living, 'Multiple floors','Only groundfloor')
View(df[,c('type_livingsqft','sqft_above','sqft_living')])
table(df$type_livingsqft) # Not concentrated around one particular type

#----------------------------------------------------------------------------------------------------------------

# Checking relationship between Dependent varibale[Price] and Independent variables.

# Price vs sqft_lot15
ggplot(df, aes(x=sqft_lot15, y=price)) + geom_point() + 
  labs(subtitle="Price vs square foot lot15", y="Price", x="square foot lot15", title="Scatterplot")
cor(df$price, df$sqft_lot15)
# 0.161

# Price vs sqft_living15
ggplot(df, aes(x=sqft_living15, y=price)) + geom_point() + 
  labs(subtitle="Price vs square foot living15", y="Price", x="square foot living15", title="Scatterplot")
cor(df$price, df$sqft_living15)
# 0.645

# Price vs sqft_above
ggplot(df, aes(x=sqft_above, y=price)) + geom_point() + 
  labs(subtitle="Price vs square foot above", y="Price", x="square foot above", title="Scatterplot")
cor(df$price, df$sqft_above)
# 0.582

# Price vs sqft_lot
ggplot(df, aes(x=sqft_lot, y=price)) + geom_point() + 
  labs(subtitle="Price vs square foot lot", y="Price", x="square foot lot", title="Scatterplot")
cor(df$price, df$sqft_lot)
# 0.146

# Price vs sqft_living
ggplot(df, aes(x=sqft_living, y=price)) + geom_point() + 
  labs(subtitle="Price vs square foot living", y="Price", x="square foot living", title="Scatterplot")
cor(df$price, df$sqft_living)
# 0.704

# Price vs sqft_basement
ggplot(df, aes(x=sqft_basement, y=price)) + geom_point() + 
  labs(subtitle="Price vs square foot basement", y="Price", x="square foot basement", title="Scatterplot")
cor(df$price, df$sqft_basement)
# 0.367 Its a low correlation mostly because of the concentration of zeros

df_non_zero_basement <- subset(df, df$sqft_basement != 0)
cor(df_non_zero_basement$price,df_non_zero_basement$sqft_basement)
# 0.428 --- moderately low
ggplot(df_non_zero_basement, aes(x=sqft_basement, y=price)) + geom_point() + 
  labs(subtitle="Price vs square foot basement", y="Price", x="square foot basement", title="Scatterplot")
ggplot(df_non_zero_basement, aes(x=sqft_basement^0.5, y=price)) + geom_point() + 
  labs(subtitle="Price vs square foot basement^0.5", y="Price", x="square foot basement", title="Scatterplot")
cor(df_non_zero_basement$price,df_non_zero_basement$sqft_basement^0.5)
# 0.384

# Checking correlation between all variables
cor_df <- df[, c("price", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15", "house_age")]
cor(cor_df)
cor_df2 <- df_non_zero_basement[, c("price", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15", "house_age")]
cor(cor_df2)

#-------------------------------------------------------------------------------------------------------------------

# Independent discrete variables and dependent variable Price
# Discrete variables -  bedrooms, bathrooms, floors, waterfront, condition, view, grade

table(df$bedrooms)
ggplot(df, aes(x=bedrooms,y=price)) + geom_bar(stat="identity", width=.10, fill="blue")

table(df$bathrooms)
# converting it into integer
df$bathrooms <- as.integer(df$bathrooms)
table(df$bathrooms)
ggplot(df, aes(x=bathrooms,y=price)) + geom_bar(stat="identity", width=.10, fill="blue")

table(df$floors)
df$floors <- as.integer(df$floors)
table(df$floors)
ggplot(df, aes(x=floors,y=price)) + geom_bar(stat="identity", width=.10, fill="blue")

table(df$condition)
ggplot(df, aes(x=condition,y=price)) + geom_bar(stat="identity", width=.5, fill="blue")

table(df$grade)
ggplot(df, aes(x=grade,y=price)) + geom_bar(stat="identity", width=.5, fill="blue")

table(df$waterfront)
table(df$view)
# Both have a lot of zeros

#---------------------------Transforming variables--------------------------------------------------------------

df <- df %>% mutate(Grades = case_when(grade %in% seq(1,3,1)  ~ "Very Low", 
                                            grade %in% seq(4,6,1)  ~ "Moderate",
                                            grade ==7              ~ "Average",
                                            grade %in% seq(8,10,1) ~ "Above Average",
                                            TRUE ~ "High"))

# Converting categorical variables to factor var.
df$Grades <- factor(df$Grades, levels = c("Very Low","Moderate","Average","Above Average","High"))
df$floors <- factor(df$floors, levels = c("1","2","3"))
df$bathrooms <- factor(df$bathrooms, levels = c("0","1","2","3","4","5"))
df$bedrooms <- factor(df$bedrooms, levels = c("0","1","2","3","4","5","6","7"))
df$is_renovated <- factor(df$is_renovated, levels = c("0","1"))
df$waterfront <- factor(df$waterfront, levels = c("0","1"))
df$view <- factor(df$view, levels = c("0","1","2","3","4"))
df$type_livingsqft <- factor(df$type_livingsqft, levels = c('Only groundfloor','Multiple floors'))
df$condition <- factor(df$condition, levels = c("1","2","3","4","5"))

cordf_1 <- df[,-c(1,12,15,16,17,18,19,26)]
names(cordf_1)

#------------------------------------------------------------Outlier Treatment--------------------------------



#Plotting Boxplots to check for outliers -  For Continuous variables

boxplot(df$price)
boxplot(df$sqft_living)
boxplot(df$sqft_lot)
boxplot(df$sqft_above)
boxplot(df$sqft_basement)
boxplot(df$sqft_living15)
boxplot(df$sqft_lot15)

#As we can see from boxplots there are lot of outliers, Let us find out the distribution of these points.
cordf_1 <- cordf_1 %>% mutate(price_ntile = ntile(price,100),
                            sqft_living_ntile = ntile(sqft_living,100),     
                            sqft_lot_ntile = ntile(sqft_lot,100),        
                            sqft_above_ntile = ntile(sqft_above,100),      
                            sqft_basement_ntile = ntile(sqft_basement,100),   
                            sqft_lot15_ntile = ntile(sqft_lot15,100),     
                            sqft_living15_ntile = ntile(sqft_living15,100)   
)

nrow(subset(cordf_1,price_ntile == 100 )) 
nrow(subset(cordf_1,sqft_living_ntile == 100 )) 
nrow(subset(cordf_1,sqft_lot_ntile == 100 )) 
nrow(subset(cordf_1,sqft_above_ntile == 100 )) 
nrow(subset(cordf_1,sqft_basement_ntile == 100 ))
nrow(subset(cordf_1,sqft_lot15_ntile == 100 )) 
nrow(subset(cordf_1,sqft_living15_ntile == 100 ))



#-------------------------------------------------------------------------------------------------------------

# We are using the meathod of capping to treat Outliers
cordf_cap <- cordf_1 %>% mutate(sqft_living_cap = ifelse(sqft_living > quantile(sqft_living,seq(0,1,0.01))[100],
                                                               quantile(sqft_living,seq(0,1,0.01))[100],sqft_living),
                                   sqft_lot_cap = ifelse(sqft_lot > quantile(sqft_lot,seq(0,1,0.01))[100],
                                                            quantile(sqft_lot,seq(0,1,0.01))[100],sqft_lot),
                                   sqft_above_cap = ifelse(sqft_above > quantile(sqft_above,seq(0,1,0.01))[100],
                                                              quantile(sqft_above,seq(0,1,0.01))[100],sqft_above),
                                   sqft_basement_cap = ifelse(sqft_basement >quantile(sqft_basement,seq(0,1,0.01))[100],
                                                                 quantile(sqft_basement,seq(0,1,0.01))[100],sqft_basement),
                                   sqft_lot15_cap = ifelse(sqft_lot15 >quantile(sqft_lot15,seq(0,1,0.01))[100],
                                                              quantile(sqft_lot15,seq(0,1,0.01))[100],sqft_lot15),
                                   sqft_living15_cap = ifelse(sqft_living15 >quantile(sqft_living15,seq(0,1,0.01))[100],
                                                                 quantile(sqft_living15,seq(0,1,0.01))[100],sqft_living15),
                                   price_cap = ifelse(price >quantile(price,seq(0,1,0.01))[100],
                                                         quantile(price,seq(0,1,0.01))[100],price)
                                   
)

#---------------------------------------------------------------------------------------------------------------


# Let us run our regression on these datasets

#1. Running our model against dataset with treated outliers.
model1_no_outliers <- lm(price_cap ~ sqft_living15_cap+sqft_lot15_cap+sqft_basement_cap+
                           sqft_above_cap+sqft_lot_cap+sqft_living_cap,
                         cordf_cap)
summary(model1_no_outliers)


#2. Running our model against dataset without treated outliers.
model1_outliers <- lm(price ~ sqft_living15+sqft_lot15+sqft_basement+
                           sqft_above+sqft_lot+sqft_living,
                         cordf)
summary(model1_outliers)


# Checking for VIF
car::vif(model1_no_outliers)
#sqft_above_cap and sqft_living_cap are giving very high VIF

# Performing a multicolinearity check
cor(sapply(cordf_cap[,c('price_cap', 'sqft_living_cap', 'sqft_lot_cap', 'sqft_above_cap', 'sqft_basement_cap', 'sqft_lot15_cap', 'sqft_living15_cap')],function(x) as.numeric(x)))


#Choosing subsets from these six variables to find out why sqft_above_cap and sqft_living_cap have such high VIF because it is important to include sqft_living_cap in the model
optimal_subset <- regsubsets(price_cap ~ sqft_living15_cap+sqft_lot15_cap+sqft_basement_cap+
                            sqft_above_cap+sqft_lot_cap+sqft_living_cap,
                          cordf_cap,
                          nvmax = 6)
summary(optimal_subset)
new_model <- summary(optimal_subset)
new_model$adjr2 #adjusted r square
new_model$bic   #BIC value
coef(optimal_subset,4)

#---------------Running regression against best 4 variables

var4_model <- lm(price_cap ~ sqft_living15_cap+sqft_basement_cap+
                   sqft_above_cap+sqft_living_cap,
                 cordf_cap)
summary(var4_model)# better Adj R squrare and reduced std. error
car::vif(var4_model) # still very high VIF for sqft_above_cap and sqft_living_cap


#-------------------------------Running regression against best 5 variables-----------------------------------------------------------------------

var5_model1 <- lm(price_cap ~ sqft_living15_cap+sqft_lot15_cap+sqft_lot_cap+sqft_basement_cap+sqft_above_cap,
                 cordf_cap)
summary(var5_model1)
car::vif(var5_model1)

var5_model <- lm(log(price_cap) ~ sqft_living15_cap+sqft_lot15_cap+sqft_lot_cap+sqft_basement_cap+sqft_above_cap,
                     cordf_cap)
summary(var5_model)
car::vif(var5_model)


#--------------------------------------------Checking for homoskedasticity
plot(var5_model1,1) #Non-homoskedasticity
plot(var5_model,1) #homoskedasticity
#Hence selecting var5_model as the best fit

#---------------------------------Normality
plot(var5_model1,2)
plot(var5_model,2)

#----------------------------------------------------------Checking for Categorical variables

# New model keeping all categorical variables and cont. vars
cat_df <- cordf_cap[,c("price_cap",
                       "bedrooms","bathrooms","floors","waterfront","view","condition",
                       "type_livingsqft","Grades","is_renovated",
                       "sqft_living_cap","sqft_lot_cap","sqft_above_cap",
                       "sqft_basement_cap","sqft_lot15_cap","sqft_living15_cap",
                       "house_age")]
ncol(cat_df)


new_catModel <- lm(price_cap ~ .,cat_df)
summary(new_catModel)
plot(new_catModel,1)

#--------------------------------------------Applying feature selection using Boruta


boruta.train <- Boruta(price_cap~.-cat_df, data = cat_df, doTrace = 2)
print(boruta.train)
plot(boruta.train, xlab = "", xaxt = "n")

#-----------------------------------------------------Let us create subsets of above datset -----------------------------------------

all_var_subset <- regsubsets(price_cap ~ .,cat_df, nvmax = 20)

data <- summary(all_var_subset)
data$adjr2
data$bic
as.data.frame(coef(all_var_subset,13))

#-----------------------------------------------------------Creating Dummy Variables

new_dummy_df <- dummy_cols(cat_df, select_columns = c("bedrooms","bathrooms","floors","waterfront","view","condition",
                                                "type_livingsqft","GradeLevels"))
names(new_dummy_df)
new_dummy_df <- new_dummy_df[,c(1,10:48)]
names(new_dummy_df)
subset_dummy <- regsubsets(price_cap ~.,new_dummy_df,nvmax = 18, really.big = F)
subset_dummy_summary <- summary(subset_dummy)
subset_dummy_summary$adjr2
subset_dummy_summary$bic
as.data.frame(coef(subset_dummy,15))


# Running our regression against these 15 variables
var15_lm <- lm(price_cap~is_renovated+sqft_living_cap+sqft_living15_cap+house_age+
                    bedrooms_6+bathrooms_4+floors_1+floors_2+view_0+
                    view_1+view_2+condition_2+condition_4+`type_livingsqft_Only groundfloor`+waterfront_1,new_dummy_df
)
summary(var15_lm)
as.data.frame(vif(var15_lm))
model.diag.metrics <- augment(var15_lm)# Skewed
plot(var15_lm,1)#heteroskedastic


# Running our regression against these 13 variables
as.data.frame(coef(subset_dummy,13))
var13_lm <- lm(price_cap~is_renovated+sqft_living_cap+sqft_living15_cap+house_age+
                 +bathrooms_4+floors_1+view_0+
                 view_1+view_2+condition_2+condition_4+`type_livingsqft_Only groundfloor`+waterfront_1,new_dummy_df
)
summary(var13_lm)
as.data.frame(vif(var13_lm)) # Better VIF
model.diag.metrics <- augment(var13_lm)
model.diag.metrics
plot(var13_lm,1)#heteroskedastic



# Running our regression against these 11 variables
as.data.frame(coef(subset_dummy,11))
var11_lm <- lm(price_cap~is_renovated+sqft_living_cap+sqft_living15_cap+house_age+
                 +bathrooms_4+floors_1+view_1+view_2+condition_2+`type_livingsqft_Only groundfloor`+waterfront_1,new_dummy_df
)
summary(var11_lm)
as.data.frame(vif(var11_lm)) # Almost Same as with 13 vars
model.diag.metrics <- augment(var11_lm)
plot(var11_lm,1)#heteroskedastic


# Selecting 12 variables model and taking log of price
var12_log_lm <- lm(log(price_cap)~is_renovated+sqft_living_cap+sqft_living15_cap+house_age+
                 +bathrooms_4+floors_1+bedrooms_2+view_1+view_2+condition_2+`type_livingsqft_Only groundfloor`+waterfront_1,new_dummy_df
)
summary(var12_log_lm)
as.data.frame(vif(var12_log_lm)) # Almost Same as with 13 vars
model.diag.metrics <- augment(var12_lm)
plot(var12_log_lm,1)# Almost homoskedastic
plot(var12_log_lm,2)
shapiro.test(var12_log_lm$residuals)
hist(var12_log_lm$residuals)# Almost Normal

wts <- 1/fitted(lm(abs(residuals(var12_log_lm)) ~ is_renovated+sqft_living_cap+sqft_living15_cap+house_age+
                     +bathrooms_4+floors_1+bedrooms_2+view_1+view_2+condition_2+`type_livingsqft_Only groundfloor`+waterfront_1,new_dummy_df))^2

model.2 <- lm(log(price_cap) ~ is_renovated+sqft_living_cap+sqft_living15_cap+house_age+
                +bathrooms_4+floors_1+bedrooms_2+view_1+view_2+condition_2+`type_livingsqft_Only groundfloor`+waterfront_1,new_dummy_df, weights=wts)

summary(model.2)
as.data.frame(vif(model.2))
plot(model.2,1)
plot(model.2,2)
hist(model.2$residuals)
shapiro.test(model.2$residuals)
# SO we have finalised this log linear model with 12 variables including weights #


#----------------------------------Let us check for interaction effects

cat_df2 <- cat_df[,c("price_cap",
                       "bedrooms","bathrooms","floors","waterfront","view","condition",
                       "type_livingsqft","Grades","is_renovated",
                       "sqft_living_cap","sqft_lot_cap","sqft_above_cap",
                       "sqft_basement_cap","sqft_lot15_cap","sqft_living15_cap",
                       "house_age")]


#-------Floors and COndition
df_intercation <- cat_df2 %>% group_by(floors, condition) %>% 
summarise(y_mean = mean(price_cap),
            y_se = psych::describe(price_cap)$se)
df_intercation %>% ggplot(aes(x = floors, y = y_mean, color = condition)) + 
  geom_line(aes(group = condition)) + geom_point() +
  labs(x = "floors",color  = "condition",y = "Dependent Variable") +
  theme_minimal() + scale_color_brewer(palette = "Dark2")

#-------------------------Grades and waterfront
df_intercation <- cat_df2 %>% group_by(Grades, waterfront) %>% 
  summarise(y_mean = mean(price_cap),
            y_se = psych::describe(price_cap)$se)
df_intercation %>% ggplot(aes(x = Grades, y = y_mean, color = waterfront)) + 
  geom_line(aes(group = waterfront)) + geom_point() +
  labs(x = "Grades",color  = "waterfront",y = "Dependent Variable") +
  theme_minimal() + scale_color_brewer(palette = "Reds")

#---------------------------------Bedroom and bathroom
df_intercation <- cat_df2 %>% group_by(bedrooms, bathrooms) %>% summarise(y_mean = mean(price_cap),
                                                                                    y_se = psych::describe(price_cap)$se)
df_intercation %>% ggplot(aes(x = bedrooms, y = y_mean, color = bathrooms)) + geom_line(aes(group = bathrooms)) +
  geom_point() + labs(x = "bedrooms",color  = "bathrooms",y = "Dependent Variable") + theme_minimal() + scale_color_brewer(palette = "Dark2")


#---------------------------------view and condition
df_intercation <- cat_df2 %>% group_by(view, condition) %>% 
  summarise(y_mean = mean(price_cap),
            y_se = psych::describe(price_cap)$se)

df_intercation %>% ggplot(aes(x = view, y = y_mean, color = condition)) + 
  geom_line(aes(group = condition)) + geom_point() +
  labs(x = "view",color  = "condition",y = "Dependent Variable") +
  theme_minimal() + scale_color_brewer(palette = "Dark2")

