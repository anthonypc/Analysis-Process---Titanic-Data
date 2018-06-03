## BAYES SOLUTION
# Information on the data set https://www.kaggle.com/c/titanic/data
# Using the data frame as per data-exploration.R

library(rstanarm)

bayes.df <- explore.df[,c(2,3,5,6,7,8,10,12)]

## Create a testing and train set
set.seed(2017)
samp <- sample(nrow(bayes.df), 0.6 * nrow(bayes.df))

t_prior <- student_t(df = 7, location = 0, scale = 2.5)
post1 <- stan_glm(Survived ~ ., data = bayes.df,
                  family = binomial(link = "logit"), 
                  prior = t_prior, prior_intercept = t_prior,
                  seed = 1)