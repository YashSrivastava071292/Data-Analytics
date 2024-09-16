#Unsupervised Learning - Individual Assignment 1
#Name: Yash Srivastava
#Email: yash_srivastava_ampba2021s@isb.edu
#PGID: 12010060


library(gtools)
library(tidyverse)  
library(cluster)
library(dplyr)
library(klustR)
library(reshape2)

#reading the given points:
a <- c(0,0)
b <- c(8,0)
c <- c(16,0)
d <- c(0,6)
e <- c(8,6)
f <- c(16,6)

vector <- c("a","b","c","d","e","f")
possible_combination <- data.frame(combinations(n=6,r=3,v=vector,repeats.allowed=F))
possible_combination # 10 cominations

df <- data.frame(rbind(a,b,c,d,e,f))
df# Dataframe from the points provided

df$NameOfPoints <- rownames(df) # Creating new column containing names
df

Results <- data.frame()
for (i in 1:nrow(possible_combination)){
  dt <- data.frame(t(possible_combination[i,]))
  names(dt) <- c("X1")
  combinations1 <- dt %>% inner_join(df, by = c("X1" = "NameOfPoints")) %>% select(X1.y, X2)
  km_results <- kmeans(df[,1:2],combinations1)
  Required <- data.frame(cbind(km_results$centers,i,km_results$iter))
  Results <- rbind(Results,Required)
}

#Generating results from above interations
names(Results) <- c("X","Y", "Combination", "Total_Iterations")

Results$XY_coordinate <- paste0( "(", Results$`X`, ",", Results$`Y`, ")")
df_final <- cbind( Results[,3:5], data.frame(rep(c("C1","C2","C3"),10)) )
names(df_final)[4] <- c("center_points")
Results <- spread(df_final, key = center_points, value = XY_coordinate )
write.csv(Results, file = "result_final.csv")
