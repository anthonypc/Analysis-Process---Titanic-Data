## Data load and transformation,
# Additional notes on requirements for the process.
library(ROCR)
library(tidyr)
library(plyr)
library(dataPreparation)
library(QuantPsyc)
library(Hmisc)
library(aod)
library(MASS)
library(ggplot2)

library(rpart)
library(DMwR)

library(mice)

library(compare)


# Information on the data set https://www.kaggle.com/c/titanic/data
fileTrain = "D:\\data\\other-projects\\data\\titanic\\train.csv"
loadTrain.file <- read.csv(fileTrain, sep = ",", quote = "\"", header = TRUE, stringsAsFactors = FALSE, na.strings=c(""," ","NA"))

fileTest = "D:\\data\\other-projects\\data\\titanic\\test.csv"
loadTest.file <- read.csv(fileTest, sep = ",", quote = "\"", header = TRUE, stringsAsFactors = FALSE, na.strings=c(""," ","NA"))

## File Check
summary(loadTrain.file)
summary(loadTest.file)

## Missingness check
sapply(loadTrain.file, function(x) {sum(is.na(x))})
sapply(loadTest.file, function(x) {sum(is.na(x))})

## Initial transformations of data.
explore.df <- loadTrain.file
explore.df$Survived <- as.factor(explore.df$Survived)
explore.df$Pclass <- as.ordered(explore.df$Pclass)
explore.df$PassengerId <- as.factor(explore.df$PassengerId)
explore.df$Cabin <- as.factor(explore.df$Cabin)
explore.df$Sex <- as.factor(explore.df$Sex)

testset.df <- loadTest.file
testset.df$Pclass <- as.ordered(testset.df$Pclass)
testset.df$PassengerId <- as.factor(testset.df$PassengerId)
testset.df$Cabin <- as.factor(testset.df$Cabin)
testset.df$Sex <- as.factor(testset.df$Sex)

## Creation of new features in the data.
explore.df$deck <- gsub("^([A-Za-z]{1}).*","\\1",explore.df$Cabin)
explore.df[which(is.na(explore.df$deck)),]$deck <- "M"
explore.df$salutation <- gsub(".*, ([A-Za-z]+)\\..*","\\1",explore.df$Name)

## Title clean up
explore.df[which(explore.df$salutation == "Jonkheer"),]$salutation <- "Sir"
explore.df[which(explore.df$salutation == "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)"),]$salutation <- "Lady"
explore.df[which(explore.df$salutation == "Ms"),]$salutation <- "Miss"
explore.df[which(explore.df$salutation == "Mme"),]$salutation <- "Mrs"
explore.df[which(explore.df$salutation == "Mlle"),]$salutation <- "Miss"
explore.df[which(explore.df$salutation == "Don"),]$salutation <- "Sir"
explore.df[which(explore.df$salutation == "Dona"),]$salutation <- "Lady"

testset.df$deck <- gsub("^([A-Za-z]{1}).*","\\1",testset.df$Cabin)
testset.df[which(is.na(testset.df$deck)),]$deck <- "M"
testset.df$salutation <- gsub(".*, ([A-Za-z]+)\\..*","\\1",testset.df$Name)
testset.df[which(testset.df$salutation == "Ms"),]$salutation <- "Miss"
testset.df[which(testset.df$salutation == "Mme"),]$salutation <- "Mrs"
testset.df[which(testset.df$salutation == "Mlle"),]$salutation <- "Miss"
testset.df[which(testset.df$salutation == "Don"),]$salutation <- "Sir"
testset.df[which(testset.df$salutation == "Dona"),]$salutation <- "Lady"

## Need to look at collapsing the high status/rare salutaions
status <- c("Dr","Lady","Rev","Col", "Sir", "Capt", "Major") ## There might be an issue here in merging M and F salutations
explore.df[which(explore.df$salutation %in% status),]$salutation <- "Status"
testset.df[which(testset.df$salutation %in% status),]$salutation <- "Status"

deck <- c("G", "T") ## There might be an issue here in merging M and F salutations
explore.df[which(explore.df$deck %in% deck),]$deck <- "GT"
testset.df[which(testset.df$deck %in% deck),]$deck <- "GT"

explore.df$salutation <- as.factor(explore.df$salutation)
explore.df$deck <- as.factor(explore.df$deck)
testset.df$salutation <- as.factor(testset.df$salutation)
testset.df$deck <- as.factor(testset.df$deck)

## Missing embarkment values
## https://www.r-bloggers.com/missing-value-treatment/
missingWip.df <- rbind(explore.df[,c(1,3:14)],testset.df)
missingWip.df$Embarked <- as.factor(missingWip.df$Embarked)
Embarked.rpart <- rpart(Embarked ~ Pclass + Sex + Age + SibSp + Parch + Fare + deck + salutation, 
                   data = missingWip.df[!is.na(missingWip.df$Embarked), ], 
                   method = "class", na.action = na.omit)
## Charting for the decision tree
par(mar = rep(0.1, 4))
plot(Embarked.rpart)
text(Embarked.rpart)
Embarked.pred <- predict(Embarked.rpart, missingWip.df)
check.df <- data.frame(actuals = as.character(missingWip.df$Embarked), predicteds = colnames(Embarked.pred)[apply(Embarked.pred, 1, which.max)])
check.df$match <- 0
check.df[which(as.character(check.df$actuals) == as.character(check.df$predicteds)),]$match <- 1

deckCheck <- table(check.df$actuals, check.df$predicteds)
deckCheck
accuracyAssess(deckCheck)

## Define the function for predicting per row
replaceEMbarked <- function(x, output){
  na.pred <- predict(Embarked.rpart, x)
  print(x)
  colnames(na.pred)[apply(na.pred, 1, which.max)]
}
## Apply the function and replace the missing values
namesEmb <- rownames(explore.df[is.na(explore.df$Embarked), ])
for(i in namesEmb){
  explore.df[i,]$Embarked <- replaceEMbarked(explore.df[i,])
}

namesEmb <- rownames(testset.df[is.na(testset.df$Embarked), ])
for(i in namesEmb){
  testset.df[i,]$Embarked <- replaceEMbarked(testset.df[i,])
}

explore.df$Embarked <- as.factor(explore.df$Embarked)
testset.df$Embarked <- as.factor(testset.df$Embarked)

## Dealing with missing age.
## Will recombinee the data as per above for the model
missingWip.df <- rbind(explore.df[,c(1,3:14)],testset.df)

age.glm <- glm(Age ~ . , data = missingWip.df[which(missingWip.df$Age > 0),c(2,4:7,9,11:12)], 
               family = gaussian(link = "identity"))
age.check <- predict(age.glm, missingWip.df, type = "response")
ageCheck <- data.frame(age.check, missingWip.df$Age)
ageCheck$diff <- ageCheck$age.check - ageCheck$missingWip.df.Age
plot(ageCheck[ageCheck$diff > 0.001 | ageCheck$diff < -0.001,]$diff)
sqrt(sum(na.omit(ageCheck$diff)^2))

age.replace <- predict(age.glm, explore.df[which(is.na(explore.df$Age)),], type = "response")
explore.df[as.numeric(names(age.replace)),]$Age <- age.replace

age.replace <- predict(age.glm, testset.df[which(is.na(testset.df$Age)),], type = "response")
testset.df[as.numeric(names(age.replace)),]$Age <- age.replace

## Need to take care of missing fares.
## This is low priority as there are not so many instances of this.
missingWip.df <- rbind(explore.df[,c(1,3:14)],testset.df)

fare.glm <- glm(Fare ~ . , data = missingWip.df[which(missingWip.df$Fare > 0),c(2,4:7,9,11:12)], 
               family = gaussian(link = "identity"))
fare.check <- predict(fare.glm, missingWip.df, type = "response")
fareCheck <- data.frame(fare.check, missingWip.df$Fare)
fareCheck$diff <- fareCheck$fare.check - fareCheck$missingWip.df.Fare
plot(fareCheck[fareCheck$diff > 0.001 | fareCheck$diff < -0.001,]$diff)
sqrt(sum(na.omit(fareCheck$diff)^2))

fare.replace <- predict(fare.glm, explore.df[which(is.na(explore.df$Fare)),], type = "response")
explore.df[as.numeric(names(fare.replace)),]$Fare <- fare.replace

fare.replace <- predict(fare.glm, testset.df[which(is.na(testset.df$Fare)),], type = "response")
testset.df[as.numeric(names(fare.replace)),]$Fare <- fare.replace

# Review of univariate relationships in the data
# Produce a correlation matrix with significance.
# These are the correlations. Factors are modified to be integers.
exploreNum.df <- colwise(as.numeric)(explore.df)
cor.dataset <- cor(exploreNum.df[sapply(exploreNum.df, is.numeric)], use = "na.or.complete", method = "pearson")

# Show raw correlations
cor.dataset
cor.prob(cor.dataset)

## Correlation plot with significance
## Will only return pairwise complete correlations. This is a little stricter than I have been taught.
corMatrix <- rcorr(as.matrix(exploreNum.df), type = "pearson")
flattenCorrMatrix(corMatrix$r, corMatrix$P)

# Correlation matrix scatter plots.
pairs(~., data = exploreNum.df, main="Simple Scatterplot Matrix")

## Univariate Checks
# Chi-square test for catagorical and t-test for continuous [still to do]
# Testing for differences within individual groups

## Reviewing the Odds ratios, confidence intervals and p values.
# Creating the formulas
univariateModels.f <- sapply(c("Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"),function(x)as.formula(paste('Survived ~',x)))

# Generate the models
univariateModels.m  <- lapply(univariateModels.f, function(x){glm(x,data = explore.df, family = "binomial")})

# Create table of Odds Ratios, Confidence Intervals and p-values for each model and each value
univariateModels.t  <- lapply(univariateModels.m,function(x){return(cbind(OddsRatio = exp(coef(x)),exp(confint(x)),pValue = coef(summary(x))[,4]))})
univariateModels.t

## Graph exploration
ggplot(explore.df, aes(x = Sex, fill = Survived)) +
  geom_bar(stat='count', position='dodge') +
  geom_label(stat='count', aes(label=..count..)) + 
  facet_grid(.~Survived)

ggplot(explore.df, aes(x = Sex, fill = Survived)) +
  geom_bar(stat='count', position='fill')

ggplot(explore.df, aes(x = Pclass, fill = Survived)) +
  geom_bar(stat='count', position='fill')

ggplot(explore.df, aes(x = Embarked, fill = Survived)) +
  geom_bar(stat='count', position='fill')

ggplot(explore.df, aes(x = SibSp, fill = Survived)) +
  geom_bar(stat='count', position='fill')

ggplot(explore.df, aes(x = Parch, fill = Survived)) +
  geom_bar(stat='count', position='fill')

ggplot(explore.df, aes(x = deck, fill = Survived)) +
  geom_bar(stat='count', position='fill')

ggplot(explore.df, aes(x = salutation, fill = Survived)) +
  geom_bar(stat='count', position='fill')

## Converting the data into a wide format.
## Will try to join the cool kids and do this as per tidyverse
## Apparently it is no illegal to not use tidyverse
## https://www.rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf

toDrop <- c("PassengerId", "Name", "Ticket", "Cabin")
toWiden <- c("Pclass", "Sex", "Embarked", "deck", "salutation")
toScale <- c("Age", "Fare")

## Drop unused columns
explore.ma <- dplyr::select(explore.df, -toDrop)
testset.ma <- dplyr::select(testset.df,-toDrop)

## Change the ordinal factor to just a factor
explore.ma$Pclass <- as.factor(as.character(explore.ma$Pclass))
testset.ma$Pclass <- as.factor(as.character(testset.ma$Pclass))

## Apply dummy variables
## Full rank
explore.ma <- fastDummies::dummy_cols(explore.ma)
testset.ma <- fastDummies::dummy_cols(testset.ma)
explore.ma <- explore.ma[,c(1,4:7,13:33)]
testset.ma <- testset.ma[,c(3:6,10:30)]

## Contrasts
explore.ma$name <- "explore.ma"
testset.ma$name <- "testset.ma"
combo.ma <- rbind(explore.ma[,-1], testset.ma)
## I am combining the two to ensure that both are using the same contrast and that both have all factor levels represented.
## Need a better method in general, but for now this is faster
combo.ma <- stats::model.matrix(~., combo.ma)

## Splitting the frames based on the name flag
explore.ma <- combo.ma[combo.ma[,"nametestset.ma"] == 0,c(1:27)]
testset.ma <- combo.ma[combo.ma[,"nametestset.ma"] == 1,c(1:27)]

#explore.ma <- stats::model.matrix(~., explore.ma)
#testset.ma <- stats::model.matrix(~., testset.ma)

## Column correction/check
Missing <- setdiff(names(explore.ma), names(testset.ma))
Missing

## Building a scale based on the training set to be applied to the test set.
scales <- dataPreparation::build_scales(dataSet = explore.ma, cols = toScale, verbose = TRUE)

## Apply standard scale to both sets of data
explore.ma <- fastScale(dataSet = explore.ma, scales = scales, verbose = TRUE)
testset.ma <- fastScale(dataSet = testset.ma, scales = scales, verbose = TRUE)

explore.ma$Survived <- explore.df$Survived
