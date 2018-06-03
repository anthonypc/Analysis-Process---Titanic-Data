## Flat model 
## Based on aggregated session and user/device tables

library(xgboost)
library(data.table)
library(e1071)
library(Matrix)
library(dplyr)
library(caret)
library(DiagrammeR)

## On to the actual modelling.
## XGBoost
## http://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html
## https://rpubs.com/mharris/multiclass_xgboost

## Initial transformations of data.
XGBoostTree.df <- explore.df[,c(2,3,5,6,7,8,10,12:14)]
XGBoostTree.df$Survived <- as.numeric(XGBoostTree.df$Survived)

## Create a testing and train set
#set.seed(2017)

## Data prep for use with xboost
## Need to explicityly transform all catagorical variables into binary

## Using a sparse matrix conversion to address this.
options(na.action='na.pass')
XGBoostTreeTrain.ma <- sparse.model.matrix(Survived~.-1, data = XGBoostTree.df[-samp, ])
outputTree_Train <- as.numeric(as.factor(XGBoostTree.df[-samp,]$Survived))-1
## Generating the xgb.DMatrix object.
## Creating the weights to be applied.
XGBoostTreeTrain.dma <- xgb.DMatrix(data = XGBoostTreeTrain.ma, label = outputTree_Train)

XGBoostTest.ma <- sparse.model.matrix(Survived~.-1, data = XGBoostTree.df[samp, ])
outputTree_Test <- as.numeric(as.factor(XGBoostTree.df[samp,]$Survived))-1
## Generating the xgb.DMatrix object.
## Creating the weights to be applied.
XGBoostTest.dma <- xgb.DMatrix(data = XGBoostTest.ma, label = outputTree_Test)

## Setting the modelling parameters
numberOfClasses <- length(unique(XGBoostTree.df$Survived))
xgbTree_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 500 # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cvTree_model <- xgb.cv(max_depth = 7, 
                  num_parallel_tree = 100, 
                   subsample = 0.5, 
                   colsample_bytree = 0.5, 
                   params = xgbTree_params,
                   data = XGBoostTreeTrain.dma,
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)

## Get the predicted status
predictedTree.xgb <- data.frame(cvTree_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = outputTree_Train + 1)
head(predictedTree.xgb)

## Assess the prediction
## Confusion table
# 1 = died  2 = survived
confusionMatrix(factor(predictedTree.xgb$label), 
                factor(predictedTree.xgb$max_prob),
                mode = "everything")

## Full model time
bstTree_model <- xgb.train(params = xgbTree_params,
                       data = XGBoostTreeTrain.dma,
                       nrounds = nround)

## Model review
model.xgb <- xgb.dump(bstTree_model, with_stats = T)
model.xgb[1:10]

# Predict hold-out test set
testTree_pred <- predict(bstTree_model, newdata = XGBoostTest.dma)
testTree_prediction <- matrix(testTree_pred, nrow = numberOfClasses,
                          ncol=length(testTree_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = outputTree_Test + 1,
         max_prob = max.col(., "last"))

# confusion matrix of test set
confusionMatrix(factor(testTree_prediction$label),
                factor(testTree_prediction$max_prob),
                mode = "everything")

# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = bstTree_model$feature_names, model = bstTree_model)
head(importance_matrix)

# plot
gp = xgb.plot.importance(importance_matrix)
print(gp) 

# Reviewing the tree
xgb.plot.tree(feature_names = bstTree_model$feature_names, model = bstTree_model, trees = 2)

## Much clearer
# http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm <- as.matrix(table(Actual = testTree_prediction$label, Predicted = factor(testTree_prediction$max_prob)))
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
