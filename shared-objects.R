## Create a testing and train set
## Create a testing and train set
set.seed(2017)
samp <- sample(nrow(explore.df), 0.6 * nrow(explore.df))
