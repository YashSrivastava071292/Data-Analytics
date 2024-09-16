#Unsupervised Learning - Individual Assignment 1
#Name: Yash Srivastava
#Email: yash_srivastava_ampba2021s@isb.edu
#PGID: 12010060




library(factoextra)
library(caret)
library(cluster)
library(dbplyr)
library(fastDummies)
library(tidyverse)
library(klustR)

# Reading the dataset
df_university <- read.csv("Universities.csv")
head(df_university)
str(df_university)

#Data Preparation:
#While performing cluster analysis, firstly we need to do data preparation.
#1. Missing values must be removed or estimated
#2. Data should be normalized.


#Removing any missing value that might be present.
sum(is.na(df_university))#2002 missing values as of now
sapply(df_university, function(x) sum(is.na(x)))# Column wise missing values
df_university_new <- na.omit(df_university)
sum(is.na(df_university_new))# verified that all missing values have been treated
nrow(df_university_new)

#Normalizing the dataset on all continous variables
z<- df_university_new[,-c(1,2,3)]
m<- apply(z, 2, mean)
s<-apply(z, 2, sd)
z <- scale(z, center=m, scale=s)
z1<- data.frame(z)
z2<- z1
nrow(z2)
str(z1)

#---------------------------------------------------------Hierarchical Clustering-----------------------------
# Dissimilarity matrix using euclidean distance
d <- dist(z1, method = "euclidean")
hcl <- hclust(d, method = "complete")# Hierarchical clustering using Complete Linkage
plot(hcl,main="Dendrogram", hang=-1)#Plotting the obtained dendrogram

# Creating subclusters using CUTREE

#Testing for 3 clusters
sub_cl1 <- cutree(hcl,k=3)
table(sub_cl1)
fviz_cluster(list(data = z1, cluster = sub_cl1))
#Summary Statistics
centers3 <- aggregate( . ~ sub_cl1, data = df_university_new[,-c(1,2,3)], FUN = mean)
centers3
summary(centers3)
dist(centers3)

#Testing for 4 clusters
sub_cl2 <- cutree(hcl,k=4)
table(sub_cl2)
fviz_cluster(list(data = z1, cluster = sub_cl2))
#Summary Statistics
centers4 <- aggregate( . ~ sub_cl2, data = df_university_new[,-c(1,2,3)], FUN = mean)
centers4
summary(centers4)
dist(centers4)


#Testing for 5 clusters
sub_cl3 <- cutree(hcl,k=5)
table(sub_cl3)
fviz_cluster(list(data = z1, cluster = sub_cl3))
#Summary Statistics
centers5 <- aggregate( . ~ sub_cl3, data = df_university_new[,-c(1,2,3)], FUN = mean)
write.csv(centers5, file="1.csv")
dist(centers5)
summary(centers5)
write.csv(summary(centers5), file = "output.csv")


#---------------------------Checking for optimal no. of clusters

#Silhouette score
plot(silhouette(cutree(hcl,3), d))#0.49
plot(silhouette(cutree(hcl,4), d))#0.49
plot(silhouette(cutree(hcl,5), d))#0.46
plot(silhouette(cutree(hcl,6), d))#0.2

# Elbow curve
fviz_nbclust(z, kmeans, method = "wss")+
  labs(subtitle = "Elbow method")

#---------------------------------Cluster stability
#To check for cluster stability we take a random 95% sample of data. Then repeating the analysis on it.

#3 clusters
set.seed(2)
df_sample <- sample(1:length(sub_cl1), length(sub_cl1)*0.05)
x <- z[-df_sample,]
dis <- dist(z[-df_sample,], method = "euclidean")
hc2 <- hclust(dis, method = "complete")
plot(hc2, hang = -1, ann = FALSE)#The dendrogram seems similar to the one before on the entire dataset.
sub_cl4 <- cutree(hc2, k=3)
table(sub_cl4)
fviz_cluster(list(data = z[-df_sample,], cluster = sub_cl4))
centers <- aggregate( . ~ sub_cl4, data = df_university_new[-df_sample, -c(1,2,3)], FUN = mean)
centers

#4 clusters
set.seed(2)
df_sample <- sample(1:length(sub_cl2), length(sub_cl2)*0.05)
x <- z[-df_sample,]
dis <- dist(z[-df_sample,], method = "euclidean")
hc2 <- hclust(dis, method = "complete")
plot(hc2, hang = -1, ann = FALSE)#The dendrogram seems similar to the one before on the entire dataset.
sub_cl4 <- cutree(hc2, k=4)
table(sub_cl4)
fviz_cluster(list(data = z[-df_sample,], cluster = sub_cl4))
centers <- aggregate( . ~ sub_cl4, data = df_university_new[-df_sample, -c(1,2,3)], FUN = mean)
centers

#5 clusters
set.seed(2)
df_sample <- sample(1:length(sub_cl3), length(sub_cl3)*0.05)
x <- z[-df_sample,]
dis <- dist(z[-df_sample,], method = "euclidean")
hc2 <- hclust(dis, method = "complete")
plot(hc2, hang = -1, ann = FALSE)#The dendrogram seems similar to the one before on the entire dataset.
sub_cl4 <- cutree(hc2, k=5)
table(sub_cl4)
fviz_cluster(list(data = z[-df_sample,], cluster = sub_cl4))
centers <- aggregate( . ~ sub_cl4, data = df_university_new[-df_sample, -c(1,2,3)], FUN = mean)
centers


#------Cluster profiling-----------------

z_new<- df_university_new[,-c(1,2,3)]
m<- apply(z_new, 2, mean)
s<-apply(z_new, 2, sd)
z_new <- scale(z_new, center=m, scale=s)
nrow(z2)
df_cluster <- df_university_new  %>% mutate(cluster = sub_cl3) 
df_cluster
row.names(z_new) <- paste(sub_cl3, ": ", df_university_new[,1], sep = "")
row.names(z_new)

heatmap(as.matrix(z_new), Colv = NA, hclustfun = hclust,  col=rev(paste("gray",1:99,sep="")))
pacoplot(data = z_new, clusters = sub_cl3,
         colorScheme = c("red", "green", "orange", "blue", "yellow"),
         labelSizes = list(yaxis = 16, yticks = 12),
         measures = list(avg = median))

#----------------------------Categorical variables
unique(df_university_new$State)
unique(df_university_new$Public..1...Private..2.)

#Converting categorical variables into dummy variables.
new_dummy_df <- dummy_cols(df_university_new, select_columns = c("State","Public..1...Private..2."))
new_dummy_df <- new_dummy_df[,-c(1,2,3)]
names(new_dummy_df)

#Repeating the same process on this dataset
sum(is.na(new_dummy_df))
new_dummy_df_norm <- sapply(new_dummy_df, scale)
new_distance <- dist(new_dummy_df_norm, method = "euclidean")
hcl1 <- hclust(new_distance, method = "complete")
plot(hcl1, cex=0.6, hang=-1)
sub_cl6<- cutree(hcl1, k=3)
centers6 <- aggregate( . ~ sub_cl6, data = new_dummy_df, FUN = mean)
centers6


#Optimal no. of clusters
plot(silhouette(cutree(hcl1,3), new_distance))#0.53
plot(silhouette(cutree(hcl1,4), new_distance))#0.19
plot(silhouette(cutree(hcl1,5), new_distance))#0.19
plot(silhouette(cutree(hcl1,6), new_distance))#0.2
fviz_nbclust(new_dummy_df_norm, kmeans, method = "wss")+
  labs(subtitle = "Elbow method")



# summary by the categorical variables
table(df_university_new$State, sub_cl1)
table(df_university_new$Public..1...Private..2., sub_cl1)
table(df_university_new$College.Name, sub_cl1)

table(df_university_new$State, sub_cl2)
table(df_university_new$Public..1...Private..2., sub_cl2)
table(df_university_new$College.Name, sub_cl2)

table(df_university_new$State, sub_cl3)
table(df_university_new$Public..1...Private..2., sub_cl3)
table(df_university_new$College.Name, sub_cl3)


#---------------------------------------------------Feature Engineering

#Acceptance ratio: application accepted/application received
df_university_new_feature <- df_university_new
df_university_new_feature<- df_university_new_feature %>% 
  mutate(accept_rate = X..appl..accepted/X..appli..rec.d*100)

df_university_new_feature<-df_university_new_feature %>% 
  mutate(total_cost = in.state.tuition+out.of.state.tuition+add..fees+estim..book.costs+estim..personal..)
df_university_new_feature

#--------------------------------Imputing missing values by using average euclidean distance.
df_university_1 <- df_university
sub<-subset(df_university_1, College.Name =="Tufts University")
sub
sum(is.na(sub))
sapply(sub, function(x) sum(is.na(x))) #X..PT.undergrad has mising value

#Taking centroid values from above
centers5 <- aggregate( . ~ sub_cl3, data = df_university_new[,-c(1,2,3)], FUN = mean)
centers5
#Calculation of euclidean distance of missing record from each of the clusters
impute_df<-centers5 %>% 
  mutate(e_dis = (((centers5$X..appli..rec.d-sub$X..appli..rec.d)^2)+((centers5$X..appl..accepted-sub$X..appl..accepted)^2)+((centers5$X..new.stud..enrolled-sub$X..new.stud..enrolled)^2)+((centers5$X..new.stud..from.top.10.- sub$X..new.stud..from.top.10.)^2)+((centers5$X..new.stud..from.top.25.-sub$X..new.stud..from.top.25.)^2)+((centers5$X..FT.undergrad-sub$X..FT.undergrad)^2)+((centers5$in.state.tuition-sub$in.state.tuition)^2)+((centers5$out.of.state.tuition-sub$out.of.state.tuition)^2)+((centers5$room-sub$room)^2)+((centers5$board-sub$board)^2)+((centers5$add..fees-sub$add..fees)^2)+((centers5$estim..book.costs-sub$estim..book.costs)^2)+((centers5$estim..personal..-sub$estim..personal..)^2)+((centers5$X..fac..w.PHD-sub$X..fac..w.PHD)^2)+((centers5$stud..fac..ratio-sub$stud..fac..ratio)^2)+((centers5$Graduation.rate-sub$Graduation.rate)^2))^0.5)
write.csv(impute_df, file="q5.csv")

#Cluster 1 is closest.
sub_cl3 <- cutree(hcl,k=5)
sub_cl3

new_df <- df_university_new %>% mutate(cluster5 = sub_cl3)
df_sub<-subset(new_df, cluster5 == 1)
value<-mean(df_sub$X..PT.undergrad)
value

sub[is.na(sub)] = value
sub