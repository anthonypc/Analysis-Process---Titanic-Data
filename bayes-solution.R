## BAYES SOLUTION
# Information on the data set https://www.kaggle.com/c/titanic/data
# Using the data frame as per data-exploration.R
# https://cran.r-project.org/web/packages/rstanarm/vignettes/rstanarm.html
## https://www.kaggle.com/avehtari/bayesian-logistic-regression-with-rstanarm

library(rstanarm)
library(shinystan )
library(LaplacesDemon)
library(ggplot2)
library(LaplacesDemon)
library(loo)

library(MASS)
library(splines)
library(bayesplot)

options(mc.cores = parallel::detectCores())

#bayes.df <- explore.df[,c(2,3,5,6,7,8,10,12)]
bayes.df <- as.data.frame(explore.ma[,c(2:26,28)])

y <- bayes.df$Survived
x <- bayes.df[,-26]

## Distribution
x <- seq(from = -5, to = 5, by = 0.1)
plot(x, LaplacesDemon::dst(x, mu = 0, sigma = 2.5, nu = 7), ylim=c(0,1), 
     type="l", main="Probability Function",
     ylab="density", col="red")

## Declaring a student prior distribution
t_prior <- student_t(df = 7, location = 0, scale = 2.5)

## Using a logit binomial and the students t prior distribution
post1 <- rstanarm::stan_glm(Survived ~ ., data = bayes.df,
                  family = binomial(link = "logit"), 
                  prior = t_prior, 
                  prior_intercept = t_prior,
                  seed = 1,
                  subset = -samp)

## Plotting 
pplot <- plot(post1, "areas", prob = 0.95, prob_outer = 1)
pplot + geom_vline(xintercept = 0)

## trace plots per variables
plot(post1, plotfun = "trace") + ggtitle("Traceplots")

## Review individual coefficients.
ci95 <- posterior_interval(post1, prob = 0.95, pars = "Sex_male")
round(ci95, 2)
ci95 <- posterior_interval(post1, prob = 0.95, pars = "Age")
round(ci95, 2)

## Reviewing the results.
round(coef(post1), 2)
round(posterior_interval(post1, prob = 0.9), 2)

## https://arxiv.org/abs/1507.04544
(loo1 <- loo(post1, save_psis = TRUE))

## Assessment against baseline
post0 <- update(post1, formula = Survived ~ 1, QR = FALSE)
(loo0 <- loo(post0))
rstanarm::compare_models(loo0,loo1)

## trace plots per variables
plot(post0, plotfun = "trace") + ggtitle("Traceplots")

## Visual assessment
launch_shinystan(post1, rstudio = getOption("shinystan.rstudio"))

## Assessing the accuracy of the insample prediction
# Predicted probabilities
yTrain <- y[-samp]
linpred <- posterior_linpred(post1)
preds <- posterior_linpred(post1, transform=TRUE)
pred <- colMeans(preds)
pr <- as.integer(pred >= 0.5)

# confusion matrix
caret::confusionMatrix(as.factor(as.numeric(pr>0.5)), yTrain)[2]
# posterior classification accuracy
round(mean(xor(pr,as.integer(yTrain==0))),2)
# posterior balanced classification accuracy
round((mean(xor(pr[yTrain==0]>0.5,as.integer(yTrain[yTrain==0])))+mean(xor(pr[yTrain==1]<0.5,as.integer(yTrain[yTrain==1]))))/2,2)

# PSIS-LOO weights
log_lik = log_lik(post1, parameter_name = "log_lik")
psis = psis(-log_lik)

#plot(psis$pareto_k)
#plot(psis$lw_smooth[,1],linpred[,1])
# LOO predictive probabilities
ploo=colSums(preds*exp(psis$lw_smooth))
# LOO classification accuracy
round(mean(xor(ploo>0.5,as.integer(yTrain==0))),2)
# LOO balanced classification accuracy
round((mean(xor(ploo[yTrain==0]>0.5,as.integer(yTrain[yTrain==0])))+mean(xor(ploo[yTrain==1]<0.5,as.integer(yTrain[yTrain==1]))))/2,2)

plot(pred,ploo)

## Checking the calebration
calPlotData <- caret::calibration(yTrain ~ pred + loopred, 
                                data = data.frame(pred=pred,loopred=ploo,y=yTrain), 
                                cuts=10, class="1")
ggplot(calPlotData, auto.key = list(columns = 2))

## 
ggplot(data = data.frame(pred=pred,loopred=ploo,y = as.numeric(yTrain)-1), aes(x=loopred, y=y)) + 
  stat_smooth(method='glm', formula = y ~ ns(x, 5), fullrange=TRUE) + 
  geom_abline(linetype = 'dashed') + ylab(label = "Observed") + xlab(label = "Predicted (LOO)") + 
  geom_jitter(height=0.03, width=0) + scale_y_continuous(breaks=seq(0,1,by=0.1)) + xlim(c(0,1))


# Predicted probability as a function of x
## Plotting for age or fare
pr_switch <- function(x, ests) plogis(ests[1] + ests[2] * x)
# A function to slightly jitter the binary data
jitt <- function(...) {
  geom_point(aes_string(...), position = position_jitter(height = 0.05, width = 0.1), 
             size = 2, shape = 21, stroke = 0.2)
}
ggplot(bayes.df[samp,], aes(x = Fare, y = as.numeric(Survived)-1, color = Survived)) + 
  scale_y_continuous(breaks = c(0, 0.5, 1)) +
  jitt(x="Age") + 
  stat_function(fun = pr_switch, args = list(ests = coef(post1)), 
                size = 2, color = "gray35")


## Horeshoe prior example
p0 <- 2 # prior guess for the number of relevant variables

n <- dim(bayes.df)[1]
p <- dim(bayes.df)[2]

tau0 <- p0/(p-p0) * 1/sqrt(n)
hs_prior <- hs(df=1, global_df=1, global_scale=tau0)
t_prior <- student_t(df = 7, location = 0, scale = 2.5)
post2 <- stan_glm(Survived ~ ., data = bayes.df,
                  family = binomial(link = "logit"), 
                  prior = hs_prior, prior_intercept = t_prior,
                  seed = 14124869, adapt_delta = 0.999,
                  subset = -samp)

pplot<-plot(post2, "areas", prob = 0.95, prob_outer = 1)
pplot + geom_vline(xintercept = 0)

(loo2 <- loo(post2))
rstanarm::compare_models(loo1,loo2)

## Checking pairwise posteriors
## If correlated can not infer variable relevance via marginal distributions
bayesplot::mcmc_pairs(as.array(post2),pars = c("Age","Fare"))

## Review insample predictions
inSample.by <- rstanarm::posterior_predict(post1)
dim(inSample.by)

inSample.byp <- posterior_linpred(post1)
dim(inSample.byp)

## Comparing the two models
par(mfrow = 1:2, mar = c(5,3.8,1,0) + 0.1, las = 3)
plot(loo1, label_points = TRUE)
plot(loo2, label_points = TRUE)

getProbabilities(post1)

## Prediction from the test set
outSample1.by <- rstanarm::posterior_predict(post1, newdata = bayes.df[samp,], type = "response", se.fit = FALSE)
table(outSample1.by)
summary(apply(outSample1.by, 1, diff))

plot(colSums(outSample1.by))
outSample1.df <- data.frame((colSums(outSample1.by)))
names(outSample1.df) <- c("Trials")

yrep_stan <- round(colMeans(outSample1.by))
table("stan_glm" = yrep_stan, "y_true" = bayes.df[samp,]$Survived)

## Review out sample predictions
outSample2.by <- rstanarm::posterior_predict(post2, newdata = bayes.df[samp,], type = "response", se.fit = FALSE)
table(outSample2.by)

plot(colSums(outSample2.by))
outSample2.df <- data.frame((colSums(outSample2.by)))
names(outSample2.df) <- c("Trials")

yrep_stan_fix <- round(colMeans(outSample2.by))
table("stan_glm" = yrep_stan_fix, "y_true" = bayes.df[samp,]$Survived)

## http://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html
cm = as.matrix(table(Actual = test.mxn.y, Predicted = yrep_stan_fix ))
cm

accuracyAssess.by <- accuracyAssess(cm)
