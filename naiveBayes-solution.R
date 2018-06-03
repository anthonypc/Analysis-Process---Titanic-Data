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

## Creating accuracy measures
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
accuracy = sum(diag) / n 
accuracy

precision = diag / colsums # fraction of correct predictions for a certain class
recall = diag / rowsums # fraction of instances of a class that were correctly predicted
f1 = 2 * precision * recall / (precision + recall) # harmonic mean (or a weighted average) of precision and recall
data.frame(precision, recall, f1) 

macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)
data.frame(macroPrecision, macroRecall, macroF1)

oneVsAll = lapply(1 : nc,
                  function(i){
                    v = c(cm[i,i],
                          rowsums[i] - cm[i,i],
                          colsums[i] - cm[i,i],
                          n-rowsums[i] - colsums[i] + cm[i,i]);
                    return(matrix(v, nrow = 2, byrow = T))})
oneVsAll

s = matrix(0, nrow = 2, ncol = 2)
for(i in 1 : nc){s = s + oneVsAll[[i]]}
s

avgAccuracy = sum(diag(s)) / sum(s)
avgAccuracy

micro_prf = (diag(s) / apply(s,1, sum))[1];
micro_prf

# Evaluation on Highly Imbalanced Datasets
# Because this is what we actually have here

mcIndex = which(rowsums==max(rowsums))[1] # majority-class index
mcAccuracy = as.numeric(p[mcIndex]) 
mcRecall = 0*p;  mcRecall[mcIndex] = 1
mcPrecision = 0*p; mcPrecision[mcIndex] = p[mcIndex]
mcF1 = 0*p; mcF1[mcIndex] = 2 * mcPrecision[mcIndex] / (mcPrecision[mcIndex] + 1)
mcIndex
mcAccuracy
# Expected accuracy for majority class.
data.frame(mcRecall, mcPrecision, mcF1) 

## Random guess
(n / nc) * matrix(rep(p, nc), nc, nc, byrow=F)
rgAccuracy = 1 / nc
rgPrecision = p
rgRecall = 0*p + 1 / nc
rgF1 = 2 * p / (nc * p + 1)
rgAccuracy
data.frame(rgPrecision, rgRecall, rgF1)

## Random weighted guesses
n * p %*% t(p)
rwgAccurcy = sum(p^2)
rwgPrecision = p
rwgRecall = p
rwgF1 = p
rwgAccurcy
data.frame(rwgPrecision, rwgRecall, rwgF1)

## Kappa
expAccuracy = sum(p*q)
kappa = (accuracy - expAccuracy) / (1 - expAccuracy)
kappa

