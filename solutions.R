

## Random Forest
## Second model
randomForestProb.df <- predict(caret_matrix, testset.df, type = "prob") # Prediction
randomForestSolution.df <- testset.df
randomForestSolution.df$Survived <- predict(caret_matrix, testset.df)
randomForestSolution.df <- randomForestSolution.df[,c("PassengerId","Survived")]

file_output("//randomForest.csv",randomForestSolution.df)

## Voting Response
votedSolution.df <- testset.df

## Random Forest
votedSolution.df$rafo <- predict(caret_matrix, votedSolution.df)

## XGBoost
voteset.df <- as.data.frame(testset.ma[,c(2:26)])
voteset.df$Survived <- 0
voteset.df$Survived <- as.factor(voteset.ma$Survived)

votetask <- makeClassifTask(data = voteset.df, target = "Survived")
votexgbt <- predict(xgtmodel, votetask)
votedSolution.df$xgb <- votexgbt$data$response

## Bayes
## Review out sample predictions
bayesVote.df <- as.data.frame(testset.ma[,c(2:26)])
outSampleVote.by <- rstanarm::posterior_predict(post2, newdata = bayesVote.df, type = "response", se.fit = FALSE)
table(outSampleVote.by)

plot(colSums(outSampleVote.by))
outSampleVote.df <- data.frame((colSums(outSampleVote.by)))
names(outSampleVote.df) <- c("Trials")

yrep_stan_vote <- round(colMeans(outSampleVote.by))
votedSolution.df$bay <- yrep_stan_vote

## majority voting and validation.
votedSolution.df$voted <- apply(votedSolution.df[,c(14:16)],1,function(x) names(which.max(table(x))))

votedSubmit.df <- votedSolution.df[,c(1,17)]
names(votedSubmit.df) <- c("PassengerId","Survived")
file_output("//voteThree.csv",votedSubmit.df)
