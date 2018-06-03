## Basic Logistic Regression Model
# Additional notes on requirements for the process.
library(ROCR)
library(plyr)
library(QuantPsyc)
library(Hmisc)
library(aod)
library(MASS)
library(ResourceSelection)
library(car)

## Create a matrix for correlations and significance
## Based on output from cor
# http://www.r-bloggers.com/more-on-exploring-correlations-in-r/
cor.prob <- function(X, dfr = nrow(X) - 2) {
  R <- cor(X)
  above <- row(R) < col(R)
  r2 <- R[above]^2
  Fstat <- r2 * dfr / (1 - r2)
  R[above] <- 1 - pf(Fstat, 1, dfr)
  R
}

## Create a matrix for correlations and significance
## Based on output from rcorr
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

## GENERALISED LOGISTIC REGRESSION PROCESS
# Multiple Binary Logistic Regression Example
# Information on the data set https://www.kaggle.com/c/titanic/data
# Using the data frame as per data-exploration.R

## Initial transformations of data.
logReg.df <- explore.df[,c(2,3,5,6,7,8,10,11,12:14)]
logReg.df$Survived <- as.factor(logReg.df$Survived)

# Review of univariate relationships in the data
# Produce a correlation matrix with significance.
# These are the correlations. Factors are modified to be integers.
processNum.df <- colwise(as.numeric)(logReg.df)
cor.dataset <- cor(processNum.df[sapply(processNum.df, is.numeric)], use = "na.or.complete", method = "pearson")


# Show raw correlations
cor.dataset
cor.prob(cor.dataset)

## Correlation plot with significance
## Will only return pairwise complete correlations. This is a little stricter than I have been taught.
corMatrix <- rcorr(as.matrix(processNum.df), type = "pearson")
flattenCorrMatrix(corMatrix$r, corMatrix$P)

# Correlation matrix scatter plots.
pairs(~., data = processNum.df, main="Simple Scatterplot Matrix")

## Univariate Checks
# Chi-square test for catagorical and t-test for continuous [still to do]
# Testing for differences within individual groups

## Reviewing the Odds ratios, confidence intervals and p values.
# Creating the formulas
univariateModels.f <- sapply(c("Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "deck", "salutation"),function(x)as.formula(paste('Survived ~',x)))

# Generate the models
univariateModels.m  <- lapply(univariateModels.f, function(x){glm(x,data = logReg.df, family = "binomial")})

# Create table of Odds Ratios, Confidence Intervals and p-values for each model and each value
univariateModels.t  <- lapply(univariateModels.m,function(x){return(cbind(OddsRatio = exp(coef(x)),exp(confint(x)),pValue = coef(summary(x))[,4]))})
univariateModels.t

## The model being generated
# Partially based on http://www.ats.ucla.edu/stat/r/dae/logit.htm
fitLog.glm <- glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+salutation, 
                  data = logReg.df, family = binomial, subset = -samp)

## Will introduce a stepwise factor selection for the model.
fitLogGLM.step <- stepAIC(fitLog.glm, trace = FALSE)
fitLogGLM.step$anova

## Stepwise optmised model
fitLogStep.glm <- glm(Survived~Pclass + Age + SibSp + Parch + Embarked + salutation, 
                  data = logReg.df, family = binomial, subset = -samp)

## Reviewing the Odds ratios, confidence intervals and p values.
# Table of values for the main model.
cbind(OddsRatio = exp(coef(fitLog.glm)), exp(confint(fitLog.glm)),pValue = coef(summary(fitLog.glm))[,4])
cbind(OddsRatio = exp(coef(fitLogStep.glm)), exp(confint(fitLogStep.glm)),pValue = coef(summary(fitLogStep.glm))[,4])

## Model attributes
summary(fitLog.glm)
summary(fitLogStep.glm)

## Reviewing the confidence intervals of estimates for features
cbind(summary(fitLog.glm)$coeff,confint(fitLog.glm), Wald = (coef(summary(fitLog.glm))[,1]/coef(summary(fitLog.glm))[,2])^2)
cbind(summary(fitLogStep.glm)$coeff,confint(fitLogStep.glm), Wald = (coef(summary(fitLogStep.glm))[,1]/coef(summary(fitLogStep.glm))[,2])^2)

## Wald Chi Square
# Overall check of the data used in the model
# Significant result indicates that there is difference between factors
wald.test(b = coef(fitLog.glm), Sigma = vcov(fitLog.glm), Terms = 2:8)
wald.test(b = coef(fitLogStep.glm), Sigma = vcov(fitLogStep.glm), Terms = 2:5)

## Collinearity Checks
# Evaluate Collinearity/Variance inflation factors 
vif(fitLog.glm) 
vif(fitLogStep.glm) 

# TRUE indicates possible issue.
sqrt(vif(fitLog.glm)) > 2
sqrt(vif(fitLogStep.glm)) > 2

## Review probabilities from the model for each row in set.
fit.glm.prob <- predict(fitLog.glm, type = "response")
fitStep.glm.prob <- predict(fitLogStep.glm, type = "response")

## Density plot of probabilities.
# Review distribution of probabilities for the whole set.
fit.glm.prob.df <- data.frame(fit.glm.prob)
ggplot(fit.glm.prob.df, aes(x = fit.glm.prob.df[,1])) + 
  geom_line(stat = "density")

fitStep.glm.prob.df <- data.frame(fitStep.glm.prob)
ggplot(fitStep.glm.prob.df, aes(x = fitStep.glm.prob.df[,1])) + 
  geom_line(stat = "density")

## Create and set contrasts.
contrasts(as.factor(processNum.df$Survived))

## Create the predictions based on a manually set threshold.
rowCount <- dim(loadTrain.file[samp,])[1]
fit.glm.pred <- rep(0, rowCount)
fit.glm.pred[fit.glm.prob > .5] = 1

resultsLog.df <- NULL
resultsLog.df$predict <- fit.glm.pred[samp]
resultsLog.df$actual <- loadTrain.file[samp,]$Survived

## Confusion table
# Check accuracy of the model
table(resultsLog.df)

## Generate predicitons for stepwise modified model
rowCount <- dim(loadTrain.file[samp,])[1]
fitStep.glm.pred <- rep(0, rowCount)
fitStep.glm.pred[fitStep.glm.prob > .5] = 1

resultsLogStep.df <- NULL
resultsLogStep.df$predict <- fitStep.glm.pred
resultsLogStep.df$actual <- loadTrain.file[samp,]$Survived

## Confusion table
# Check accuracy of the model
table(resultsLogStep.df)

## Check models against training data.
classificationError <- mean(fit.glm.pred != loadTrain.file[samp,]$Survived)
print(paste('Accuracy',1-classificationError))

classificationError <- mean(fitStep.glm.pred != loadTrain.file[samp,]$Survived)
print(paste('Accuracy',1-classificationError))

## Hosmer and Lemeshow test of goodness of fit.
# Significant result indicates that the model is not a good fit.
hoslem.test(loadTrain.file[samp,]$Survived, fit.glm.pred, g = 10)
hoslem.test(loadTrain.file[samp,]$Survived, fitStep.glm.pred, g = 10)

## Create a ROC for the mmodel.
fit.glm.pred <- predict(fitLog.glm, loadTrain.file[samp,], type = "response")
fitStep.glm.pred <- predict(fitLogStep.glm, loadTrain.file[samp,], type = "response")

fit.glm.pr <- prediction(fit.glm.pred, loadTrain.file$Survived)
fitStep.glm.pr <- prediction(fitStep.glm.pred, loadTrain.file$Survived)

fit.glm.cur <- performance(fit.glm.pr, measure = "tpr", x.measure = "fpr")
plot(fit.glm.cur)
fitStep.glm.cur <- performance(fitStep.glm.pr, measure = "tpr", x.measure = "fpr")
plot(fitStep.glm.cur)

fit.aucd <- performance(fit.glm.pr, measure = "auc")
fit.aucd <- fit.aucd@y.values[[1]]
fit.aucd

fitStep.aucd <- performance(fitStep.glm.pr, measure = "auc")
fitStep.aucd <- fitStep.aucd@y.values[[1]]
fitStep.aucd


## Review probabilities from the model for each row in set.
fitStep.glm <- predict(fitLogStep.glm, newdata = logReg.df[samp,])

## http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm = as.matrix(table(Actual = resultsLogStep.df$actual , Predicted = resultsLogStep.df$predict ))

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

accuracy.log = sum(diag) / n 
accuracy.log 

precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
prere.log <- data.frame(precision, recall, f1) 
prere.log

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
