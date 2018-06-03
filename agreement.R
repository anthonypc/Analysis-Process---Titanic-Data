## Checking the agreement between the two models.
## Actual Data
compare.df <- explore.df[,c(2,3,5,6,7,8,10,12)]

nbpred <- predicted.nb
length(nbpred)

svmpred <- svmCheck.df$Pred
length(svmpred)

logpred <- resultsLogStep.df$predict
length(logpred)

rfpred <- resultsOne.df$predict
length(rfpred)

xgbpred <- test_prediction$max_prob - 1
length(xgbpred)

agreement <- data.frame(actual = compare.df[samp,]$Survived, svmpred = svmpred, nbpred = nbpred, logpred = logpred, rfpred = rfpred, xgbpred = xgbpred)

compareSamp.df <- cbind(compare.df[samp,], agreement[2:5])

## majority voting and validation.
compareSamp.df$voted <- apply(agreement,1,function(x) names(which.max(table(x))))

## Comparing the models.
## https://stats.stackexchange.com/questions/28523/how-to-get-percentage-agreement-between-a-group-of-factor-columns
y <- apply(compareSamp.df[,c(1,9:13)], 2, function(x) factor(x, levels=c("0", "1")))
mmult <- function(f=`*`, g=sum) 
  function(x, y) apply(y, 2, function(a) apply(x, 1, function(b) g(f(a,b))))

`%**%` <- mmult(`==`, mean)

t(y) %**% y 


## Much clearer
# http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm <- as.matrix(table(Actual = voted.df$actual, Predicted = voted.df$voted))
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