## RANDOM FOREST SOLUTION
# Information on the data set https://www.kaggle.com/c/titanic/data
# Using the data frame as per data-exploration.R

library(randomForest)
library(caret)
library(pROC)

randomForest.df <- explore.df[,c(2,3,5,6,7,8,10,12)]
randomForest.df$Survived <- as.ordered(randomForest.df$Survived)

## Create a testing and train set
#set.seed(2017)
#samp <- sample(nrow(randomForest.df), 0.6 * nrow(randomForest.df))

## Address the missing values in the data
if(dim(randomForest.df[is.na(randomForest.df),])[1] > 0){
  randomForestNA.df <- rfImpute(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = randomForest.df)
} else {
  randomForestNA.df <- randomForest.df
}

## Returning the results of the analysis with the new set
randomForest.bag <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = randomForestNA.df, subset = -samp, mtry = 7, importance = TRUE)
randomForest.bag

yhat.bag <- predict (randomForest.bag, newdata = randomForestNA.df[samp,]) 
resultsOne.df <- NULL
resultsOne.df$predict <- yhat.bag
resultsOne.df$actual <- randomForestNA.df[samp,]$Survived
table(resultsOne.df)

## An alternative approach
caret_matrix <- train(x = randomForestNA.df[,c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")], 
                      y = randomForestNA.df$Survived, 
                      data = randomForestNA.df, 
                      method = 'rf', 
                      trControl = trainControl(method = "cv", number = 5))
caret_matrix
caret_matrix$results

varImpPlot(caret_matrix$finalModel, main=" Variable importance")

solution_rf <- predict(caret_matrix, randomForestNA.df[samp,])
resultsTwo.df <- NULL
resultsTwo.df$predict <- solution_rf
resultsTwo.df$actual <- randomForestNA.df$Survived
table(resultsTwo.df)

## Create a ROC for the model.
rf.roc <- roc(randomForestNA.df[samp,]$Survived, randomForest.bag$votes[,2])
plot(rf.roc)
auc(rf.roc)

## Second model
result.predicted.prob <- predict(caret_matrix, randomForestNA.df[samp,], type = "prob") # Prediction

names(result.predicted.prob) <- c("Died", "Survived")
result.roc <- roc(as.ordered(randomForestNA.df$Survived), result.predicted.prob$Survived) # Draw ROC curve.
plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")

result.coords <- coords(result.roc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))
print(result.coords)#to get threshold and accuracy

## http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm = as.matrix(table(Actual = resultsOne.df$actual , Predicted = resultsOne.df$predict ))

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

accuracy.rf = sum(diag) / n 
accuracy.rf 

precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
prere.rf <- data.frame(precision, recall, f1) 
prere.rf

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
