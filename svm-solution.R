## SVM SOLUTION
# Information on the data set https://www.kaggle.com/c/titanic/data
# Using the data frame as per data-exploration.R

library(e1071)
library(gmodels)
library(caret)

svm.df <- explore.df[,c(2,3,5,6,7,8,10,12:14)]

## Create a testing and train set
#set.seed(2017)
#samp <- sample(nrow(svm.df), 0.6 * nrow(svm.df))

svm.df$rowNumber <- rownames(svm.df)
svm.df$rowNumber <- as.numeric(svm.df$rowNumber)

########
## And again with SVM
## Constructing the simple model.
weights01 <- table(svm.df$Survived)  # the weight vector must be named with the classes names
weights01[1] <- 1    # a class +1 mismatch not so much...
weights01[2] <- 1 # a class -1 mismatch has a terrible cost
weights01

## Modeling tuning.
#svm.tune <- tune(svm, Converted ~ ., data = trainingTerms, kernel = "radial", ranges = list(cost=10^(-1:2), gamma =  c(.5,1,2)))
#print(svm.tune)

model01.svm <- svm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + salutation, data = svm.df, cost = 0.1, gamma = 0.5, probability = TRUE, class.weights = weights01, subset = -samp)
predict01.svm <- predict(model01.svm, svm.df[samp,], decision.values = FALSE)

## Cross table of the model's results.
svmCheck.df <- NULL
svmCheck.df$Survived <- svm.df[samp,]$Survived
svmCheck.df <- as.data.frame(svmCheck.df)
svmCheck.df$Pred <- NA
predict01.df <- as.data.frame(predict01.svm)
#svmCheck.df[rownames(predict01.df), ]$Pred <- predict01.df$predict01.svm
svmCheck.df$Pred <- predict01.df$predict01.svm
CrossTable(svmCheck.df$Pred, svmCheck.df$Survived, prop.chisq = TRUE, prop.t = FALSE, dnn = c("predicted", "actual"))

predict01.svmProb <- predict(model01.svm, svm.df[-samp,], decision.values = TRUE, probability = TRUE)

## Probability of conversion by row.
model01Prob.svm <- data.frame(attr(predict01.svmProb, "probabilities"))
model01Prob.svm$rowNumber <- rownames(model01Prob.svm)
model01Prob.svm$rowNumber <- as.numeric(model01Prob.svm$rowNumber)

rowProbs.svm <- merge(svm.df[-samp,], model01Prob.svm,  by = "rowNumber", all.x = TRUE)
rowProbs.svm$Pred <- svmCheck.df$Pred

## Create a ROC for the mmodel.
pr.svm <- prediction(model01Prob.svm$X1, svm.df[rownames(predict01.df), "Survived"])
prf.svm <- performance(pr.svm, measure = "tpr", x.measure = "fpr")
plot(prf.svm)

auc.svm <- performance(pr.svm, measure = "auc")
auc.svm <- auc.svm@y.values[[1]]
auc.svm

## Overall Accuracy figure
## This is a little sensitive to inputs regarding the threashold at which low frequency observations are removed.
tab <- table(predict01.svm, svm.df[rownames(predict01.df), "Survived"])
sum(tab[row(tab)==col(tab)])/sum(tab)

postResample(predict(model01.svm, svm.df), svm.df[rownames(predict01.df),]$Survived)

## http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm = as.matrix(table(Actual = svmCheck.df$Survived, Predicted = svmCheck.df$Pred))

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

accuracy = sum(diag) / n 
accuracy 

precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
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

