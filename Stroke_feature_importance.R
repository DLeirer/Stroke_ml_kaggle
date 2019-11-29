
# title: "Stroke Feature Importance"
# author: "Daniel Leirer"
# date: "10/08/2019"

## Housekeeping Notes

#This document is a short report detailing my analysis of the Stroke data from kaggle (https://www.kaggle.com/asaumya/healthcare-dataset-stroke-data).

#It should come as part of a zip file containing the original data, the models I generated, and a script that will allow replication of all results seen here. 
#All results were generated on a Dell Latitude E7470, running Windows 8.1 and RStudio.


# Load Libraries ----------------------------------------------------------
library(tidyverse)
library(dplyr)
library(caret)
library(DataExplorer)




# Functions ---------------------------------------------------------------


# function to loop through all machine learning models.
ml_fun<- function(dataset) {
  # GLMNET
  set.seed(seed)
  fit.glmnet <- train(Phenotype~., data=dataset, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
  print("Glmnet Done")
  # SVM Radial
  set.seed(seed)
  fit.svmRadial <- train(Phenotype~., data=dataset, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
  print("SVM Radial Done")
  # SVM Linear
  set.seed(seed)
  fit.svmLinear <- train(Phenotype~., data=dataset, method="svmLinear2", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
  print("SVM Linear Done")
  # kNN
  set.seed(seed)
  fit.knn <- train(Phenotype~., data=dataset, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
  print("KNN Done")
  # Naive Bayes
  set.seed(seed)
  fit.nb <- train(Phenotype~., data=dataset, method="nb", metric=metric, trControl=control)
  print("NB Done")
  # Random Forest
  set.seed(seed)
  fit.rf <- train(Phenotype~., data=dataset, method="rf", metric=metric, trControl=control)
  print("RF Done")
  ##Make List of models
  modellist<-list(glmnet=fit.glmnet, svmRad=fit.svmRadial, svmLinear=fit.svmLinear, knn=fit.knn, nb=fit.nb, rf=fit.rf)
  modellist
}



# I did not write this function. I makes a pretty nice confusion matrix plot. 
draw_confusion_matrix <- function(cm) {
  
  total <- sum(cm$table)
  res <- as.numeric(cm$table)
  
  # Generate color gradients. Palettes come from RColorBrewer.
  greenPalette <- c("#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B")
  redPalette <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A","#EF3B2C","#CB181D","#A50F15","#67000D")
  getColor <- function (greenOrRed = "green", amount = 0) {
    if (amount == 0)
      return("#FFFFFF")
    palette <- greenPalette
    if (greenOrRed == "red")
      palette <- redPalette
    colorRampPalette(palette)(100)[10 + ceiling(90 * amount / total)]
  }
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col=getColor("green", res[1]))
  text(195, 435, classes[1], cex=1.2)
  rect(250, 430, 340, 370, col=getColor("red", res[3]))
  text(295, 435, classes[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col=getColor("red", res[2]))
  rect(250, 305, 340, 365, col=getColor("green", res[4]))
  text(140, 400, classes[1], cex=1.2, srt=90)
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}

# Define Directories ------------------------------------------------------

#working directory
top_dir = "DEFINE DIRECTORY"



#data directory
data_dir = "./data/"

#output directory
output_dir = "./output/"


# Load data ---------------------------------------------------------------

#set working directory
setwd(top_dir)

#define csv name
data_name = "train_2v.csv"

#create path to data
data_path = paste(data_dir,data_name,sep ="")

#load csv
train_2v = read.csv(data_path)


# Explore data ------------------------------------------------------------

# 1 check data loaded
str(train_2v)

#plot str
plot_str(train_2v)


# summary of data.
summary(train_2v)


# A few things jump out right away. 
# Gender = 11 other?  Transgender or something else. 
# Age minimum is 0.08.  
# Normal glucose levels should be 72 and 140. We see down to 55 and up to 291. Individuals might have diabetes. 
# Bmi looks reasonable allthough BMI of 97 is suspiciously high.

# plot intro
plot_intro(train_2v)

# plot missing
plot_missing(train_2v)


# Check ID's no ID's are duplicated. 
table(duplicated(train_2v$id))


# 2 univariate plots

#### Plot bar graphs for data. 
plot_bar(train_2v) 

plot_bar(train_2v[is.na(train_2v$bmi),])

#looks like smoking status contains an NA category that I missed previously. Needs to be changed. 
table(train_2v$smoking_status)

#Change smoking status to unknown. Missing data might contain valuable information. 
levels(train_2v$smoking_status)[1] <- "unknown" 

#double check smoking status
table(train_2v$smoking_status)


#### Plot historgrams 
plot_histogram(train_2v)

# age looks okay. a bit choppy, not quite sure why. Very young range is odd. 
table(train_2v$age)

#glucose level has two peaks. the second probably being diabetes. potential for feature engineering. 
#BMI looks fine. A few outliers maybe at the extreme. But that will probably have little impact. 


#qq plot
plot_qq(train_2v[,c("age","bmi","avg_glucose_level")], sampled_rows = 1000L)




# 3 bivariable plots
plot_correlation(na.omit(train_2v))


#continues variables correlation plot. 
plot_correlation(na.omit(train_2v),type="c")

#discreet variable correlation plot. 
plot_correlation(na.omit( train_2v[,c("gender","ever_married","work_type","smoking_status","Phenotype")]))


### Boxplot Stroke vs continues variables.

# change stroke column to factor
train_2v <- train_2v %>% mutate(Phenotype = ifelse(stroke == 1, "Stroke","Healthy"))
train_2v$Phenotype = factor(train_2v$Phenotype,levels(as.factor(train_2v$Phenotype))[c(2,1)])



# make Boxplots
plot_boxplot(train_2v[,c("age","avg_glucose_level","bmi","Phenotype")], by = "Phenotype")

# Age seems to be a major factor, as does glucose. 






# Split Data --------------------------------------------------------------

# Turn Hypertension and heart disease into factors
train_2v <- train_2v %>% mutate(hypertension = ifelse(hypertension == 0, "no_hypertension","hypertension"))
train_2v$hypertension = as.factor(train_2v$hypertension)
train_2v <- train_2v %>% mutate(heart_disease = ifelse(heart_disease == 0, "no_heart_disease","heart_disease"))
train_2v$heart_disease = as.factor(train_2v$heart_disease)

#Remove "Gender Other"
train_2v = train_2v %>%  filter(gender != "Other")
train_2v$gender=droplevels(train_2v$gender)
  
# drop NAs
train_2v=train_2v[!is.na(train_2v$bmi),]

# make id row names
rownames(train_2v) = train_2v$id

# drop old id and stroke columns
train_2v$id = NULL
train_2v$stroke = NULL


#Set Seed
seed = 7
set.seed(seed)

#Create training and testing df
training_index <- createDataPartition(train_2v$Phenotype, p = 0.70, list = F)

training_df <- train_2v[training_index,]
testing_df <- train_2v[-training_index,]



# Spot check algorithms ---------------------------------------------------

#Control for algorithms
control <- trainControl(method="repeatedcv", number=5, repeats=3, sampling="down", savePredictions="final",classProbs = T,summaryFunction = twoClassSummary)

#Define Metric to choose best model
metric <- "ROC"


### Warning may use a lot of resources and take a long time to run. Output is a list of machine learning models. 
## I uncommented the next two lines because you can just load the models. 
#ml_models<-ml_fun(training_df)

#Save Models
#save(ml_models, file=paste(output_dir,"Classification_models.rdata",sep=""), compress = T)

#Load models
load(file=paste(output_dir,"Classification_models.rdata",sep=""))

#### plot model resampling results. 

#Resampling data from models
ml_resample<-resamples(ml_models)
#Take ROC metrics from df. 
resample_metric<-as.data.frame(ml_resample,metric = ml_resample$metrics[1])

# reshape resampling data.
resample_results<-gather(resample_metric,"Resample")
colnames(resample_results) = c("Algorithm","AUC")

#plot resampling metrics for all models to compare. 
ggplot(resample_results,
       aes(x=Algorithm,y=AUC,color=Algorithm)) +
  geom_boxplot()+
  geom_point()+ 
  ggtitle("3x5-fold cross validation for 6 Models")






# Check glmnet model in more detail ---------------------------------------


# take glmnet model from model list. 
fit.glmnet = ml_models$glmnet


# predict testing data using glmnet model
predictions <- predict(fit.glmnet, testing_df)


#make confusion matrix. 
cm<-confusionMatrix(predictions, testing_df$Phenotype)

#draw confusion matrix. 
draw_confusion_matrix(cm)



# feature importance ------------------------------------------------------



#Feature Importance glmnet
importance <- varImp(fit.glmnet, scale=FALSE)
plot(importance)


#Feature Importance RF
importance_rf <- varImp(ml_models$rf, scale=FALSE)
plot(importance_rf)



# Glmnet probability predictions and exploratory plots --------------------


#calculate probabilty predictions
predictions_prob <- predict(fit.glmnet, testing_df,type = "prob")


#add probability prediction to testing df
testing_df<-cbind(testing_df,predictions_prob)

# plots using probability predictions. 
ggplot(testing_df,
       aes(x=heart_disease,y=Stroke,color=Phenotype)) +
  geom_boxplot(notch = T)+
  ylab("Probability score for Stroke")+
  ggtitle("Stroke probability vs Heart Disease (GLMNET Model)")

ggplot(testing_df,
       aes(x=hypertension,y=Stroke,color=Phenotype)) +
  geom_boxplot(notch = T)+
  ylab("Probability score for Stroke")+
  ggtitle("Stroke probability vs Hypertension (GLMNET Model)")

ggplot(testing_df,
       aes(x=smoking_status,y=Stroke,color=Phenotype)) +
  geom_boxplot(notch = T)+
  ylab("Probability score for Stroke")+
  ggtitle("Stroke probability vs smoking status (GLMNET Model)")
