## BAYES SOLUTION
# Information on the data set https://www.kaggle.com/c/titanic/data
# Using the data frame as per data-exploration.R
# https://cran.r-project.org/web/packages/rstanarm/vignettes/rstanarm.html

library(rstanarm)
library(shinystan )
library(LaplacesDemon)
library(ggplot2)
library(LaplacesDemon)

options(mc.cores = parallel::detectCores())

#bayes.df <- explore.df[,c(2,3,5,6,7,8,10,12)]
bayes.df <- explore.ma[,c(2:26,28)]

## Distribution
x <- seq(from = -5, to = 5, by = 0.1)
plot(x, LaplacesDemon::dst(x, mu = 0, sigma = 2.5, nu = 7), ylim=c(0,1), 
     type="l", main="Probability Function",
     ylab="density", col="red")

## Declaring a student prior distribution
t_prior <- student_t(df = 7, location = 0, scale = 2.5)

## Using a logit binomial and the students t prior distribution
post1 <- stan_glm(Survived ~ ., data = bayes.df,
                  family = binomial(link = "logit"), 
                  prior = t_prior, 
                  prior_intercept = t_prior,
                  seed = 1,
                  subset = -samp)

## Plotting 
pplot <- plot(post1, "areas", prob = 0.95, prob_outer = 1)
pplot + geom_vline(xintercept = 0)

## Review individual coefficients.
ci95 <- posterior_interval(post1, prob = 0.95, pars = "Sex_male")
round(ci95, 2)
ci95 <- posterior_interval(post1, prob = 0.95, pars = "Age")
round(ci95, 2)

## Reviewing the results.
round(coef(post1), 2)
round(posterior_interval(post1, prob = 0.9), 2)

launch_shinystan(post1, rstudio = getOption("shinystan.rstudio"))

## Review insample predictions
inSample.by <- rstanarm::posterior_predict(post1)
dim(inSample.by)

inSample.byp <- posterior_linpred(post1)
dim(inSample.byp)

log_lik(post1)

####
inSample.df <- bayes.df[-samp,]
inSample.df$Survived <- as.numeric(inSample.df$Survived)
inSample.df$Survived <- inSample.df$Survived - 1

inSample.byp.df <- as.data.frame(inSample.byp)
users.byp.df <- t(inSample.byp.df)
users.byp.df$users <- as.numeric(rownames(users.byp.df))



boxplot(inSample.byp.df[inSample.df$Survived == "1"])
with(inSample.df, points(rownames(inSample.df), Survived, pch = 16, col = "red"))





boxplot(sweep(inSample.byp, 2, STATS = 
                as.numeric(inSample.df$Survived), FUN = "/"), 
        axes = FALSE, main = "Male", pch = NA,
        xlab = "Years of Education", ylab = "Proportion of Agrees")
with(inSample.df, axis(1, at = deck_M))






loo_bglm_1 <- loo(post1)
plot(loo_bglm_1, label_points = TRUE)





## Review out sample predictions
outSample.by <- rstanarm::posterior_predict(post1, newdata = bayes.df[samp,], type = "response", se.fit = FALSE)

