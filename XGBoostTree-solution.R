## Flat model 
## Based on aggregated session and user/device tables

library(xgboost)
library(data.table)
library(e1071)
library(Matrix)
library(dplyr)
library(caret)
library(DiagrammeR)

# For the learner
library(mlr)
library(parallelMap) 

## On to the actual modelling.
## XGBoost
## http://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html
## https://cran.r-project.org/web/packages/xgboost/vignettes/xgboostPresentation.html
## https://rpubs.com/mharris/multiclass_xgboost
## https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/

## Notes on the technique
# Extension of GBM, it is a part of the boosted gradiant family
# Typically can out perform GBM
# Different in its pruning methods (max depth than prune, inbuilt cross validation, and other)

## Initial transformations of data.
XGBoostTree.df <- explore.df[,c(2,3,5,6,7,8,10,12:14)]
XGBoostTree.df$Survived <- as.numeric(XGBoostTree.df$Survived)

## Create a testing and train set
#set.seed(2017)

## Data prep for use with xgboost
## Need to explicityly transform all catagorical variables into binary

## Using a sparse matrix conversion to address this.
options(na.action='na.pass')
XGBoostTrainTree.ma <- sparse.model.matrix(Survived~.-1, data = XGBoostTree.df[-samp, ])
outputTree_Train <- as.numeric(as.factor(XGBoostTree.df[-samp,]$Survived))-1
## Generating the xgb.DMatrix object.
## Creating the weights to be applied.
XGBoostTrainTree.dma <- xgb.DMatrix(data = XGBoostTrainTree.ma, label = outputTree_Train)

XGBoostTestTree.ma <- sparse.model.matrix(Survived~.-1, data = XGBoostTree.df[samp, ])
outputTree_Test <- as.numeric(as.factor(XGBoostTree.df[samp,]$Survived))-1
## Generating the xgb.DMatrix object.
## Creating the weights to be applied.
XGBoostTestTree.dma <- xgb.DMatrix(data = XGBoostTestTree.ma, label = outputTree_Test)

## Setting the modelling parameters
## These are specific for trees. For linear regression a different set would be used.
numberOfClasses <- length(unique(XGBoostTree.df$Survived))
xgb_paramsTree <- list(booster = "gbtree",
                   eta = 0.3,
                   gamma = 0,
                   num_parallel_tree = 100, 
                   max_depth = 3,
                   min_child_weight = 1,
                   subsample = 1, 
                   colsample_bytree = 1)
nround    <- 500 # number of XGBoost rounds
cv.nfold  <- 5 # 
## Address the imbalanced classes
survived_cases <- length(XGBoostTree.df[which(XGBoostTree.df$Survived == 2),]$Survived)
deceased_cases <- length(XGBoostTree.df[which(XGBoostTree.df$Survived == 1),]$Survived)
scale_pos_weight = survived_cases/deceased_cases

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cvTree_model <- xgb.cv(params = xgb_paramsTree,
                   data = XGBoostTrainTree.dma,
                   objective = "binary:logistic",
                   nrounds = nround,
                   nfold = cv.nfold,
                   prediction = TRUE, 
                   showsd = T, 
                   stratified = T, 
                   print_every_n = 10, 
                   early_stop_rounds = 20, 
                   maximize = F,
                   scale_pos_weight = scale_pos_weight)

## Get the predicted status
predictedTree.xgb <- data.frame(cvTree_model$pred)
names(predictedTree.xgb) <- "prediction"
predictedTree.xgb$max_prob <- 0
predictedTree.xgb[which(predictedTree.xgb$prediction > 0.5),]$max_prob <- 1
predictedTree.xgb$label <- outputTree_Train

## Assess the prediction
## Confusion table
# 1 = died  2 = survived
confusionMatrix(factor(predictedTree.xgb$label), 
                factor(predictedTree.xgb$max_prob),
                mode = "everything")

## Full model time
## Check for the existance of the tuned parameters, if these exist use these
## Otherwise just use the training set
if(typeof(mytuneTree$x) == "list") { train_paramsTree <- mytuneTree$x } else { train_paramsTree <- xgb_paramsTree }
watchlist <- list(train = XGBoostTrainTree.dma, test = XGBoostTestTree.dma)
## Run gtraining
bstTree_model <- xgb.train(params = train_paramsTree,
                           objective = "binary:logistic", 
                           eval_metric = "error",
                       data = XGBoostTrainTree.dma,
                       nrounds = nround,
                       print_every_n = 10, 
                       watchlist = watchlist,
                       scale_pos_weight = scale_pos_weight)

## Model review
label = getinfo(XGBoostTestTree.dma, "label")
pred <- predict(bstTree_model, XGBoostTestTree.dma)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))

modelTree.xgb <- xgb.dump(bstTree_model, with_stats = T)
modelTree.xgb[1:10]

xgb.plot.deepness(model = bstTree_model)

# Predict hold-out test set
testTree_pred <- predict(bstTree_model, newdata = XGBoostTestTree.dma)
predictedTestTree.xgb <- data.frame(testTree_pred)
names(predictedTestTree.xgb) <- "prediction"
predictedTestTree.xgb$max_prob <- 0
predictedTestTree.xgb[which(predictedTestTree.xgb$prediction > 0.5),]$max_prob <- 1
predictedTestTree.xgb$label <- outputTree_Test

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

## Adding a part here for further optimisation
# Create tasks
XGBoostTreeMLR.df <- XGBoostTree.df
XGBoostTreeMLR.df$Survived <- as.factor(XGBoostTreeMLR.df$Survived)
fact_col <- colnames(XGBoostTreeMLR.df)[sapply(XGBoostTreeMLR.df,is.character)]
for(i in fact_col) set(XGBoostTreeMLR.df,j=i,value = factor(XGBoostTreeMLR.df[[i]]))
# One hot encoding
XGBoostTreeMLR.df <- createDummyFeatures (obj = XGBoostTreeMLR.df, target = "Survived")
# Creation of the tasks
traintask <- makeClassifTask (data = XGBoostTreeMLR.df[-samp, ], target = "Survived")
testtask <- makeClassifTask (data = XGBoostTreeMLR.df[samp, ], target = "Survived")

#create learner
lrn <- makeLearner("classif.xgboost",
                   predict.type = "response")
lrn$par.vals <- list( objective = "multi:softprob", 
                      eval_metric = "mlogloss", 
                      nrounds = 100L, 
                      eta = 0.1)
#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",
                                          values = c("gbtree","gblinear")), 
                        makeNumericParam("eta",
                                         lower = 0.1, upper = 1),
                        makeIntegerParam("num_parallel_tree",
                                         lower = 10L, upper = 200L), 
                        makeNumericParam("gamma",
                                         lower = 0, upper = 1),
                        makeIntegerParam("max_depth",
                                         lower = 3L, upper = 10L), 
                        makeNumericParam("min_child_weight",
                                         lower = 1L, upper = 10L), 
                        makeNumericParam("subsample",
                                         lower = 0.5, upper = 1), 
                        makeNumericParam("colsample_bytree",
                                         lower = 0.5, upper = 1))
#set resampling strategy
rdesc <- makeResampleDesc("CV", stratify = T, iters=5L)
#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytuneTree <- tuneParams(learner = lrn, 
                     task = traintask, 
                     resampling = rdesc, 
                     measures = acc, 
                     par.set = params, 
                     control = ctrl, 
                     show.info = T)

#set hyperparameters
lrn_tune <- setHyperPars(lrn, par.vals = mytuneTree$x)
#train model
xgmodel <- train(learner = lrn_tune, task = traintask)
#predict model
xgpred <- predict(xgmodel, testtask)
## Assess the prediction
## Confusion table
# 1 = died  2 = survived
confusionMatrix(xgpred$data$response,
                xgpred$data$truth,
                mode = "everything")
## Adjusted parametres
mytuneTree$x

## Much clearer
# http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm <- as.matrix(table(Actual = predictedTestTree.xgb$label, Predicted = factor(predictedTestTree.xgb$max_prob)))
cm

## Creating accuracy measures
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
accuracy.xgbt = sum(diag) / n 
accuracy.xgbt

precision = diag / colsums # fraction of correct predictions for a certain class
recall = diag / rowsums # fraction of instances of a class that were correctly predicted
f1 = 2 * precision * recall / (precision + recall) # harmonic mean (or a weighted average) of precision and recall
prere.xgbt <- data.frame(precision, recall, f1) 
prere.xgbt

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
