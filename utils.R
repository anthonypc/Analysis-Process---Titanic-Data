## Utilities files
## Load functions.

## Create a testing and train set
## Create a testing and train set
set.seed(2017)
samp <- sample(nrow(explore.df), 0.6 * nrow(explore.df))


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

