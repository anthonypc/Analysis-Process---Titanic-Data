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

xgbpredt <- xgpred$data$response
length(xgbpredt)

xgbttpred <- xgpredt$data$response
length(xgbttpred)

xgbtpred <- predictedTestTree.xgb$max_prob
length(xgbtpred)

gbmpred <- solution_boost
length(gbmpred)

mxnetfpred <- predff.label
length(mxnetfpred)

agreement <- data.frame(actual = compare.df[samp,]$Survived, 
                        svmpred = svmpred, 
                        nbpred = nbpred, 
                        logpred = logpred, 
                        rfpred = rfpred, 
                        xgbpred = xgbpred, 
                        xgbpredt = xgbpredt,
                        xgbtpred = xgbtpred, 
                        xgbttpred = xgbttpred, 
                        gbmpred = gbmpred,
                        mxnetfpred = mxnetfpred)

compareSamp.df <- cbind(compare.df[samp,], agreement[2:9])

## majority voting and validation.
compareSamp.df$voted <- apply(agreement[,c(2,3,5,6,7,8)],1,function(x) names(which.max(table(x))))

## Comparing the models.
## https://stats.stackexchange.com/questions/28523/how-to-get-percentage-agreement-between-a-group-of-factor-columns
y <- apply(compareSamp.df[,c(1,9:16)], 2, function(x) factor(x, levels=c("0", "1")))
mmult <- function(f=`*`, g=sum) 
  function(x, y) apply(y, 2, function(a) apply(x, 1, function(b) g(f(a,b))))

`%**%` <- mmult(`==`, mean)

t(y) %**% y 


## Much clearer
# http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm <- as.matrix(table(Actual = compareSamp.df$Survived, Predicted = compareSamp.df$voted))
cm

accuracyAssess.ag <- accuracyAssess(cm)

## Model Reviews
## Collected accuracy measures

accuracyAssess.ag$accuracy
accuracyAssess.nb$accuracy
accuracyAssess.svm$accuracy
accuracyAssess.rf$accuracy
accuracyAssess.log$accuracy
accuracyAssess.xgb$accuracy
accuracyAssesst.xgb$accuracy
accuracyAssess.xgbt$accuracy
accuracyAssess.xgbtt$accuracy
accuracyAssess.gbm$accuracy
accuracyAssess.mxnf$accuracy

accuracyAssess.ag$PrecRecf1
accuracyAssess.nb$PrecRecf1
accuracyAssess.svm$PrecRecf1
accuracyAssess.rf$PrecRecf1
accuracyAssess.log$PrecRecf1
accuracyAssess.xgb$PrecRecf1
accuracyAssesst.xgb$PrecRecf1
accuracyAssess.xgbt$PrecRecf1
accuracyAssess.xgbtt$PrecRecf1
accuracyAssess.gbm$PrecRecf1
accuracyAssess.mxnf$PrecRecf1

accuracyAssess.ag$kappa
accuracyAssess.nb$kappa
accuracyAssess.svm$kappa
accuracyAssess.rf$kappa
accuracyAssess.log$kappa
accuracyAssess.xgb$kappa
accuracyAssesst.xgb$kappa
accuracyAssess.xgbt$kappa
accuracyAssess.xgbtt$kappa
accuracyAssess.gbm$kappa
accuracyAssess.mxnf$kappa
