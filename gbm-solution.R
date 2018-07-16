## Stochastic Gradient Boosting Tree model
## Based on aggregated session and user/device tables
library(gbm)
library(caret)
library(pROC)

## On to the actual modelling.
## Gradient Boosted Model

## Initial transformations of data.
gbm.df <- explore.df[,c(2,3,5,6,7,8,10,12:14)]
gbm.df$Survived <- as.factor(gbm.df$Survived)

caret_boost <- train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                     data = gbm.df, 
                     method = 'gbm', 
                     preProcess = c('center', 'scale'), 
                     trControl = trainControl(method="cv", number=7), 
                     verbose = FALSE,
                     subset = -samp)
print(caret_boost)

solution_boost <- predict(caret_boost, gbm.df[samp,])

## Create a ROC for the model.
gbm.roc <- roc(as.ordered(gbm.df[samp,]$Survived), as.ordered(solution_boost))
plot(gbm.roc)
auc(gbm.roc)


## http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm = as.matrix(table(Actual = gbm.df[samp,]$Survived, Predicted = solution_boost ))

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

accuracy.gbm = sum(diag) / n 
accuracy.gbm 

precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
prere.gbm <- data.frame(precision, recall, f1) 
prere.gbm

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

mcIndex = which(rowsums==max(rowsums))[1] # majority-class index
mcAccuracy = as.numeric(p[mcIndex]) 
mcRecall = 0*p;  mcRecall[mcIndex] = 1
mcPrecision = 0*p; mcPrecision[mcIndex] = p[mcIndex]
mcF1 = 0*p; mcF1[mcIndex] = 2 * mcPrecision[mcIndex] / (mcPrecision[mcIndex] + 1)
mcIndex
mcAccuracy
data.frame(mcRecall, mcPrecision, mcF1) 

(n / nc) * matrix(rep(p, nc), nc, nc, byrow=F)
rgAccuracy = 1 / nc
rgPrecision = p
rgRecall = 0*p + 1 / nc
rgF1 = 2 * p / (nc * p + 1)
rgAccuracy
data.frame(rgPrecision, rgRecall, rgF1)

n * p %*% t(p)
rwgAccurcy = sum(p^2)
rwgPrecision = p
rwgRecall = p
rwgF1 = p
rwgAccurcy
data.frame(rwgPrecision, rwgRecall, rwgF1)

expAccuracy = sum(p*q)
kappa = (accuracy - expAccuracy) / (1 - expAccuracy)
kappa



