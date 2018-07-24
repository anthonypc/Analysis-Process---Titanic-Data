## Flat model 
## Based on aggregated session and user/device tables

library(caret)
library(data.table)
library(e1071)

## On to the actual modelling.
## Naive Bayes

## load-explore
naiveBayes.df <- explore.df[,c(2,3,5,6,7,8,10,12:14)]

## Create a testing and train set
## Create a testing and train set
#set.seed(2017)
#samp <- sample(nrow(naiveBayes.df), 0.6 * nrow(naiveBayes.df))

## Generate the model
userAnwers.nb <- naiveBayes(Survived ~ ., data = naiveBayes.df, subset = -samp, 
                            na.action = na.pass, laplace = 1)
## Diagnostics
#summary(userAnwers.nb)
#userAnwers.nb

predicted.nb <- predict(userAnwers.nb, naiveBayes.df[samp, ])

## Assess the prediction
## Confusion table
table(naiveBayes.df[samp, ]$Survived)
table(predicted.nb)
table(predicted.nb, naiveBayes.df[samp, ]$Survived) # predicted = x

## Much clearer
# http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm <- as.matrix(table(Actual = naiveBayes.df[samp, ]$Survived, Predicted = predicted.nb))
cm

accuracyAssess.nb <- accuracyAssess(cm)
