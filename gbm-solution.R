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
cm

accuracyAssess.gbm <- accuracyAssess(cm)
