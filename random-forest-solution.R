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
cm

accuracyAssess.rf <- accuracyAssess(cm)